#!/usr/bin/env python3
"""Stage 20: how much do the headline comparisons depend on the one train/test split?

Every comparison in the paper rests on a single 70/30 split of METABRIC, and the
differences between methods are small: the test-set C-indices of the clustering
procedures lie between 0.533 and 0.565. Variation from one random split to another
could easily be larger than that, in which case a comparison on one split says little.

This script repeats the whole evaluation over several independent splits. Each split
is drawn with the same stratification as the original (vital status crossed with the
PAM50-surrogate label, rare strata pooled), and every data-dependent step is refitted
on that split's training patients only: the low-variance gene filter that determines
the ssGSEA input, the recurrent-mutation gene list, the view scalers, MOFA, the choice
of k, the k-means centroids, the Cox models and the clinical imputation. Nothing fitted
on one split is reused on another. The copy-number pathway and arm features contain no
fitted component and are computed once.

On each split the following are scored on that split's held-out patients:
    survival-guided k (the pipeline evaluated in the paper)
    silhouette-selected k on the same MOFA factors
    survival-guided k on the concatenated features, without MOFA
    SNF, and the NEMO, CIMLR and iClusterPlus approximations, at the survival-guided k
    clinical variables, PAM50 surrogate, and clusters plus clinical variables

Repeated random splits share most of their training data, so the per-split differences
are positively correlated and an ordinary paired t-test across splits overstates the
evidence. Comparisons are therefore made with the corrected resampled t-test of Nadeau
and Bengio (2003), which inflates the variance by a factor (1/J + n_test/n_train).

Outputs:
    outputs/tables/repeated_splits_per_split.csv
    outputs/tables/repeated_splits_summary.csv
    outputs/tables/repeated_splits_comparisons.csv
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import gseapy as gp
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import load_config, get_logger, resolve


def _load(name):
    path = Path(__file__).parent / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stratified_split(clin, test_fraction, seed):
    event = clin["OS_EVENT"].astype(int).astype(str)
    subtype = clin["CLAUDIN_SUBTYPE"].fillna("Unknown").replace("", "Unknown")
    strat = event + "_" + subtype.astype(str)
    counts = strat.value_counts()
    rare = counts[counts < 4].index
    strat = strat.where(~strat.isin(rare), other=event + "_RARE")
    tr, te = train_test_split(clin.index, test_size=test_fraction, stratify=strat, random_state=seed)
    return pd.Index(tr), pd.Index(te)


def ssgsea_scores(mrna, gene_sets, cfg):
    s = cfg["feature_aggregation"]["ssgsea"]
    batch = s.get("sample_batch_size") or mrna.shape[0]
    ids = list(mrna.index)
    chunks = []
    for i in range(0, len(ids), batch):
        expr = mrna.loc[ids[i:i + batch]].T
        expr.index.name = "Gene"
        res = gp.ssgsea(data=expr, gene_sets=gene_sets, outdir=None,
                        sample_norm_method=s["sample_norm_method"],
                        min_size=s["min_geneset_size"], max_size=s["max_geneset_size"],
                        threads=s["threads"], no_plot=True)
        chunks.append(res.res2d.pivot(index="Name", columns="Term", values="NES").astype(float))
        del res, expr
        gc.collect()
    return pd.concat(chunks, axis=0).reindex(mrna.index)


def survival_k(Z, dur, ev, ccfg, logrank_test):
    scores = {}
    for k in range(ccfg["k_min"], ccfg["k_max"] + 1):
        rep = []
        for r in range(ccfg["n_inner_repeats"]):
            seed = ccfg["random_seed"] + 1000 * k + r
            tr, va = train_test_split(np.arange(len(Z)), test_size=ccfg["inner_val_fraction"],
                                      stratify=ev, random_state=seed)
            km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=seed).fit(Z[tr])
            lab = km.predict(Z[va])
            if len(np.unique(lab)) < 2:
                rep.append(-np.inf); continue
            p = logrank_test(dur[va], lab, ev[va]).p_value
            rep.append(-np.log10(max(p, 1e-300)))
        scores[k] = float(np.mean(rep))
    return max(scores, key=scores.get)


def corrected_resampled_t(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on J paired differences."""
    d = np.asarray(d, dtype=float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), df=J - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("20_repeated_outer_splits", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    genesets_dir = resolve(cfg["paths"]["genesets_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ccfg, mcfg = cfg["clustering"], cfg["mofa"]
    horizons = cfg["evaluation"]["time_dependent_auc_horizons_months"]

    s03 = _load("03_feature_aggregation.py")
    s06 = _load("06_cluster_evaluation.py")
    s10 = _load("10_baseline_comparison.py")     # also applies the snfpy compatibility shim
    s18 = _load("18_geneset_seed_sensitivity.py")
    import snf
    from lifelines.statistics import multivariate_logrank_test

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    covariates = pd.read_parquet(proc_dir / "clinical_covariates.parquet")
    mrna = pd.read_parquet(proc_dir / "mrna_raw.parquet")
    cna = pd.read_parquet(proc_dir / "cna_raw.parquet")
    mut_bin = pd.read_parquet(proc_dir / "mutation_binary.parquet")

    # gene sets are the paper's: same collections, same subsampling seed
    gene_sets = s03.build_gene_sets(cfg, genesets_dir, logger)
    arm_map = s03.chrom_arm_map(genesets_dir / "hgnc_complete_set.txt")
    cna_view = pd.concat([s03.cna_arm_burden(cna, arm_map), s03.cna_pathway_scores(cna, gene_sets)], axis=1)

    mcfg_top = cfg["feature_aggregation"]["mutations"]
    order = mcfg["views"]
    rows = []

    for j in range(args.splits):
        seed = 9000 + j
        tr_ids, te_ids = stratified_split(clinical, cfg["split"]["test_fraction"], seed)
        dur_tr, ev_tr = clinical.loc[tr_ids, "OS_MONTHS_NUM"].values, clinical.loc[tr_ids, "OS_EVENT"].values
        dur_te, ev_te = clinical.loc[te_ids, "OS_MONTHS_NUM"].values, clinical.loc[te_ids, "OS_EVENT"].values

        # --- features: every fitted step refitted on this split's training patients ---
        var = mrna.loc[tr_ids].var(axis=0)
        keep = var[var >= np.percentile(var, cfg["preprocess"]["min_gene_variance_percentile"])].index
        mrna_view = ssgsea_scores(mrna[keep], gene_sets, cfg)

        rec = mut_bin.loc[tr_ids].sum(axis=0)
        rec = rec[rec >= cfg["preprocess"]["mutation_min_recurrence"]]
        top = rec.sort_values(ascending=False).head(mcfg_top["recurrent_gene_list_top_n"]).index
        mut_view = mut_bin[top].copy()
        mut_view.columns = [f"MUT__{g}" for g in top]
        mut_view["TMB"] = mut_bin.sum(axis=1)

        views = {"mrna_pathway": mrna_view, "cna_pathway": cna_view, "mutation": mut_view}
        scalers = {v: StandardScaler().fit(views[v].loc[tr_ids].values) for v in order}
        std_tr = {v: scalers[v].transform(views[v].loc[tr_ids].values) for v in order}
        std_te = {v: scalers[v].transform(views[v].loc[te_ids].values) for v in order}

        Z_tr, Z_te = s18.fit_mofa_project(std_tr, std_te, order, mcfg, seed_offset=500 + j)

        res = {"split": j + 1, "seed": seed, "n_train": len(tr_ids), "n_test": len(te_ids),
               "n_factors": Z_tr.shape[1]}

        # --- survival-guided k on MOFA factors (the evaluated pipeline) ---
        k_s = survival_k(Z_tr, dur_tr, ev_tr, ccfg, multivariate_logrank_test)
        km = KMeans(n_clusters=k_s, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(Z_tr)
        lab_tr, lab_te = km.labels_, km.predict(Z_te)
        c, p, _ = s10.cox_eval(lab_tr, lab_te, dur_tr, ev_tr, dur_te, ev_te)
        res.update({"k_survival": k_s, "c_survival": c, "p_survival": p})

        # --- silhouette-selected k on the same factors ---
        sil = {k: silhouette_score(Z_tr, KMeans(n_clusters=k, n_init=20, random_state=ccfg["random_seed"]).fit_predict(Z_tr))
               for k in range(ccfg["k_min"], ccfg["k_max"] + 1)}
        k_sil = max(sil, key=sil.get)
        km2 = KMeans(n_clusters=k_sil, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(Z_tr)
        c, p, _ = s10.cox_eval(km2.labels_, km2.predict(Z_te), dur_tr, ev_tr, dur_te, ev_te)
        res.update({"k_silhouette": k_sil, "c_silhouette": c, "p_silhouette": p})

        # --- no MOFA: survival-guided k on the concatenated standardised features ---
        X_tr = np.hstack([std_tr[v] for v in order]); X_te = np.hstack([std_te[v] for v in order])
        k_nm = survival_k(X_tr, dur_tr, ev_tr, ccfg, multivariate_logrank_test)
        km3 = KMeans(n_clusters=k_nm, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(X_tr)
        c, p, _ = s10.cox_eval(km3.labels_, km3.predict(X_te), dur_tr, ev_tr, dur_te, ev_te)
        res.update({"k_no_mofa": k_nm, "c_no_mofa": c, "p_no_mofa": p})

        # --- alternative procedures at the survival-guided k, under the common inductive rule ---
        aff = snf.make_affinity([std_tr[v] for v in order], metric="sqeuclidean", K=20, mu=0.5)
        lab = SpectralClustering(n_clusters=k_s, affinity="precomputed",
                                 random_state=ccfg["random_seed"]).fit_predict(snf.snf(aff, K=20))
        c, p, _ = s10.cox_eval(lab, s10.nearest_centroid_assign(X_tr, lab, X_te), dur_tr, ev_tr, dur_te, ev_te)
        res.update({"c_snf": c, "p_snf": p})

        lab = SpectralClustering(n_clusters=k_s, affinity="precomputed", random_state=ccfg["random_seed"]).fit_predict(
            np.mean([s10.self_tuned_affinity(std_tr[v]) for v in order], axis=0))
        c, p, _ = s10.cox_eval(lab, s10.nearest_centroid_assign(X_tr, lab, X_te), dur_tr, ev_tr, dur_te, ev_te)
        res.update({"c_nemo": c, "p_nemo": p})

        lab = SpectralClustering(n_clusters=k_s, affinity="precomputed", random_state=ccfg["random_seed"]).fit_predict(
            np.mean([s10.multi_kernel_affinity(std_tr[v]) for v in order], axis=0))
        c, p, _ = s10.cox_eval(lab, s10.nearest_centroid_assign(X_tr, lab, X_te), dur_tr, ev_tr, dur_te, ev_te)
        res.update({"c_cimlr": c, "p_cimlr": p})

        pca = PCA(n_components=Z_tr.shape[1], random_state=ccfg["random_seed"]).fit(X_tr)
        km4 = KMeans(n_clusters=k_s, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(pca.transform(X_tr))
        c, p, _ = s10.cox_eval(km4.labels_, km4.predict(pca.transform(X_te)), dur_tr, ev_tr, dur_te, ev_te)
        res.update({"c_icluster": c, "p_icluster": p})

        # --- clinical comparators, fitted on this split's training patients ---
        clin_tr, clin_te = s06.prep_covariates(covariates, tr_ids, te_ids)
        pam_tr, pam_te = s06.pam50_dummies(covariates, tr_ids, te_ids)
        cd_tr = pd.get_dummies(pd.Series(lab_tr, index=tr_ids), prefix="cluster", drop_first=True).astype(float)
        cd_te = pd.get_dummies(pd.Series(lab_te, index=te_ids), prefix="cluster", drop_first=True).reindex(
            columns=cd_tr.columns, fill_value=0.0).astype(float)
        for name, (a, b) in {"clinical": (clin_tr, clin_te), "pam50": (pam_tr, pam_te),
                             "cluster_plus_clinical": (pd.concat([cd_tr, clin_tr], axis=1),
                                                       pd.concat([cd_te, clin_te], axis=1))}.items():
            r = s06.fit_cox_evaluate(a, dur_tr, ev_tr, b, dur_te, ev_te, horizons, logger, name)
            res[f"c_{name}"] = r["c_index"]

        rows.append(res)
        logger.info("Split %2d/%d: k=%d (silhouette %d, no-MOFA %d) | C-index survival-k %.3f, "
                    "silhouette %.3f, no-MOFA %.3f, SNF %.3f, NEMO %.3f, CIMLR %.3f, iCluster %.3f, "
                    "clinical %.3f", j + 1, args.splits, k_s, k_sil, k_nm, res["c_survival"],
                    res["c_silhouette"], res["c_no_mofa"], res["c_snf"], res["c_nemo"],
                    res["c_cimlr"], res["c_icluster"], res["c_clinical"])
        del mrna_view, views, std_tr, std_te, Z_tr, Z_te, X_tr, X_te
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "repeated_splits_per_split.csv", index=False)

    methods = {"survival": "Survival-guided k (evaluated pipeline)", "silhouette": "Silhouette-selected k",
               "no_mofa": "Survival-guided k, no MOFA", "snf": "SNF", "nemo": "NEMO approximation",
               "cimlr": "CIMLR approximation", "icluster": "iClusterPlus approximation",
               "pam50": "PAM50 surrogate", "clinical": "Clinical variables",
               "cluster_plus_clinical": "Clusters plus clinical"}
    summ = [{"method": lab, "mean_c_index": df[f"c_{m}"].mean(), "sd_c_index": df[f"c_{m}"].std(ddof=1),
             "min_c_index": df[f"c_{m}"].min(), "max_c_index": df[f"c_{m}"].max()}
            for m, lab in methods.items()]
    pd.DataFrame(summ).to_csv(tables_dir / "repeated_splits_summary.csv", index=False)

    n_tr, n_te = df.n_train.iloc[0], df.n_test.iloc[0]
    comp = []
    for m, lab in methods.items():
        if m == "survival":
            continue
        d = df["c_survival"] - df[f"c_{m}"]
        t_naive, p_naive = stats.ttest_1samp(d, 0.0)
        t_c, p_c = corrected_resampled_t(d, n_tr, n_te)
        comp.append({"comparison": f"Survival-guided k vs {lab}", "mean_difference": d.mean(),
                     "sd_difference": d.std(ddof=1), "splits_survival_better": int((d > 0).sum()),
                     "n_splits": len(d), "p_naive_paired_t": p_naive, "p_corrected_resampled_t": p_c})
    pd.DataFrame(comp).to_csv(tables_dir / "repeated_splits_comparisons.csv", index=False)

    logger.info("")
    logger.info("k selected by the survival criterion across splits: %s", df.k_survival.tolist())
    for s in summ:
        logger.info("%-42s mean C-index %.3f (sd %.3f, range %.3f-%.3f)", s["method"],
                    s["mean_c_index"], s["sd_c_index"], s["min_c_index"], s["max_c_index"])
    for c in comp:
        logger.info("%-62s diff %+.3f, better in %d/%d, corrected p %.3f", c["comparison"],
                    c["mean_difference"], c["splits_survival_better"], c["n_splits"],
                    c["p_corrected_resampled_t"])
    logger.info("Stage 20 complete.")


if __name__ == "__main__":
    main()
