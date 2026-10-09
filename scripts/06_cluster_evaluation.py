#!/usr/bin/env python3
"""Stage 06: rigorous evaluation of the survival-aware clusters.

1. Prognostic performance on the held-out TEST set: concordance index (C-index)
   and time-dependent AUC, benchmarked against clinical variables alone and
   against PAM50 (CLAUDIN_SUBTYPE surrogate) alone, plus a combined model.
2. Cluster stability: subsample the TRAINING set 10x (80% each), rerun MOFA
   + k-means (k fixed at the value selected in stage 05) on each subsample,
   and compute the adjusted Rand index against the reference train
   clustering restricted to the subsampled patients. (ssGSEA pathway/CNA/
   mutation feature computation is a deterministic per-sample transform with
   no cross-sample fitting, so it is not re-run per subsample -- only the
   data-dependent steps, MOFA factorisation and k-means, are, which is what
   "stability" concerns.)
3. Biological characterisation, descriptive only: for every mRNA-pathway and
   CNA feature, the standardised mean difference between each cluster and
   the rest of the training set; for every recurrent mutation, the mutation
   frequency inside and outside each cluster. No hypothesis tests are run
   here. The clusters were built from these same variables, so a test of
   whether the variables differ between clusters on the same patients has
   no inferential meaning, and reporting one invites it to be read as if it
   did. The summaries describe the partition; they do not test it.

Outputs (outputs/tables/):
    survival_performance_test.csv
    cluster_stability_ari.csv
    cluster_biological_pathways.csv
    cluster_biological_mutations.csv
data/processed/metabric/clinical_with_clusters.parquet (merged, for stage 09)
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from common import load_config, get_logger, resolve

VIEW_FILES = {
    "mrna_pathway": "view_mrna_pathway.parquet",
    "cna_pathway": "view_cna_pathway.parquet",
    "mutation": "view_mutation.parquet",
}


def prep_covariates(covariates: pd.DataFrame, train_ids, test_ids):
    numeric_cols = ["AGE_AT_DIAGNOSIS", "GRADE", "TUMOR_SIZE", "TUMOR_STAGE", "LYMPH_NODES_EXAMINED_POSITIVE"]
    cat_cols = ["ER_IHC", "HER2_SNP6", "PR_STATUS" if "PR_STATUS" in covariates.columns else None,
                "INFERRED_MENOPAUSAL_STATE"]
    cat_cols = [c for c in cat_cols if c]

    num = covariates[numeric_cols].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median").fit(num.loc[train_ids])
    num_imp = pd.DataFrame(imputer.transform(num), index=num.index, columns=numeric_cols)

    cat = covariates[cat_cols].fillna("Unknown").replace("", "Unknown")
    cat_dummies = pd.get_dummies(cat, drop_first=True)

    X = pd.concat([num_imp, cat_dummies], axis=1).astype(float)
    return X.loc[train_ids], X.loc[test_ids]


def pam50_dummies(covariates: pd.DataFrame, train_ids, test_ids):
    subtype = covariates["CLAUDIN_SUBTYPE"].fillna("Unknown")
    dummies = pd.get_dummies(subtype, prefix="PAM50", drop_first=True).astype(float)
    return dummies.loc[train_ids], dummies.loc[test_ids]


def fit_cox_evaluate(X_train, dur_train, ev_train, X_test, dur_test, ev_test, horizons, logger, name):
    df_train = X_train.copy()
    df_train["duration"], df_train["event"] = dur_train, ev_train
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(df_train, duration_col="duration", event_col="event")
    except Exception as e:
        logger.warning("Cox fit failed for model '%s' (%s); returning NaNs", name, e)
        return {"model": name, "c_index": np.nan, **{f"auc_{h}m": np.nan for h in horizons}}

    risk_test = cph.predict_partial_hazard(X_test).values.ravel()
    c_index = concordance_index(dur_test, -risk_test, ev_test)

    y_train_s = Surv.from_arrays(event=ev_train.astype(bool), time=dur_train)
    y_test_s = Surv.from_arrays(event=ev_test.astype(bool), time=dur_test)
    result = {"model": name, "c_index": c_index}
    valid_horizons = [h for h in horizons if h < dur_test.max()]
    if valid_horizons:
        try:
            auc_vals, _ = cumulative_dynamic_auc(y_train_s, y_test_s, risk_test, valid_horizons)
            for h, a in zip(valid_horizons, auc_vals):
                result[f"auc_{h}m"] = a
        except Exception as e:
            logger.warning("Time-dependent AUC failed for '%s' (%s)", name, e)
    for h in horizons:
        result.setdefault(f"auc_{h}m", np.nan)
    return result


def refit_mofa_cluster_subsample(proc_dir, subsample_ids, view_order, mcfg, k, kmeans_seed, n_init):
    views_std = {}
    for view in view_order:
        df = pd.read_parquet(proc_dir / VIEW_FILES[view]).loc[subsample_ids]
        views_std[view] = StandardScaler().fit_transform(df.values)

    feat_names = [[f"{v}_{i}" for i in range(views_std[v].shape[1])] for v in view_order]
    data = [[views_std[v]] for v in view_order]
    ep = entry_point()
    ep.set_data_options(scale_views=False, center_groups=False)
    ep.set_data_matrix(data, likelihoods=["gaussian"] * len(view_order), views_names=view_order,
                        groups_names=["g"], samples_names=[subsample_ids], features_names=feat_names)
    ep.set_model_options(factors=mcfg["n_factors_default"])
    ep.set_train_options(iter=mcfg["max_iterations"], convergence_mode=mcfg["convergence_mode"],
                          dropR2=mcfg["drop_factor_threshold"], seed=kmeans_seed, quiet=True, verbose=False)
    ep.build()
    ep.run()
    Z = ep.model.nodes["Z"].getExpectation()
    labels = KMeans(n_clusters=k, n_init=n_init, random_state=kmeans_seed).fit_predict(Z)
    return pd.Series(labels, index=subsample_ids)


def describe_features(feat_df: pd.DataFrame, labels: pd.Series, prefix: str) -> pd.DataFrame:
    """Per-cluster standardised mean difference (cluster vs rest, pooled SD) for every feature.

    Descriptive: these numbers characterise the partition and are not tests of it.
    """
    rows = []
    labels = labels.loc[feat_df.index]
    for cluster in sorted(labels.unique()):
        in_mask = (labels == cluster).values
        x_in, x_out = feat_df.loc[in_mask], feat_df.loc[~in_mask]
        pooled_sd = np.sqrt((x_in.var(ddof=1) + x_out.var(ddof=1)) / 2).replace(0, np.nan)
        smd = (x_in.mean() - x_out.mean()) / pooled_sd
        for col in feat_df.columns:
            rows.append({"cluster": cluster, "feature": col, "mean_in": x_in[col].mean(),
                         "mean_out": x_out[col].mean(), "smd": smd[col],
                         "direction": "up" if smd[col] > 0 else "down"})
    df = pd.DataFrame(rows)
    df["abs_smd"] = df["smd"].abs()
    df["source"] = prefix
    return df.sort_values(["cluster", "abs_smd"], ascending=[True, False])


def describe_mutations(mut_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mutation frequency inside vs outside each cluster for every recurrent gene (descriptive)."""
    labels = labels.loc[mut_df.index]
    rows = []
    for cluster in sorted(labels.unique()):
        in_mask = (labels == cluster).values
        n_in, n_out = int(in_mask.sum()), int((~in_mask).sum())
        for col in mut_df.columns:
            a = int(((mut_df[col] == 1) & in_mask).sum())
            b = int(((mut_df[col] == 1) & ~in_mask).sum())
            freq_in, freq_out = a / n_in, b / n_out
            rows.append({"cluster": cluster, "gene": col.replace("MUT__", ""),
                         "n_mutated_in_cluster": a, "n_in_cluster": n_in,
                         "n_mutated_outside": b, "n_outside": n_out,
                         "freq_in_cluster": freq_in, "freq_outside": freq_out,
                         "freq_ratio": freq_in / freq_out if freq_out > 0 else np.nan})
    df = pd.DataFrame(rows)
    df["abs_freq_diff"] = (df["freq_in_cluster"] - df["freq_outside"]).abs()
    return df.sort_values(["cluster", "abs_freq_diff"], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("06_cluster_evaluation", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    covariates = pd.read_parquet(proc_dir / "clinical_covariates.parquet")
    cl_train = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    cl_test = pd.read_parquet(proc_dir / "cluster_labels_test.parquet")["cluster"]

    train_ids, test_ids = cl_train.index, cl_test.index
    dur_train, ev_train = clinical.loc[train_ids, "OS_MONTHS_NUM"].values, clinical.loc[train_ids, "OS_EVENT"].values
    dur_test, ev_test = clinical.loc[test_ids, "OS_MONTHS_NUM"].values, clinical.loc[test_ids, "OS_EVENT"].values
    horizons = cfg["evaluation"]["time_dependent_auc_horizons_months"]

    # ---- 1. survival performance benchmarking on held-out test set -----------------------
    cluster_dummies_train = pd.get_dummies(cl_train, prefix="cluster", drop_first=True).astype(float)
    cluster_dummies_test = pd.get_dummies(cl_test, prefix="cluster", drop_first=True).reindex(
        columns=cluster_dummies_train.columns, fill_value=0.0).astype(float)

    clin_train, clin_test = prep_covariates(covariates, train_ids, test_ids)
    pam50_train, pam50_test = pam50_dummies(covariates, train_ids, test_ids)

    models = {
        "cluster_only": (cluster_dummies_train, cluster_dummies_test),
        "clinical_only": (clin_train, clin_test),
        "pam50_only": (pam50_train, pam50_test),
        "cluster_plus_clinical": (pd.concat([cluster_dummies_train, clin_train], axis=1),
                                   pd.concat([cluster_dummies_test, clin_test], axis=1)),
    }
    perf_rows = []
    for name, (Xtr, Xte) in models.items():
        logger.info("Fitting benchmark Cox model: %s", name)
        res = fit_cox_evaluate(Xtr, dur_train, ev_train, Xte, dur_test, ev_test, horizons, logger, name)
        perf_rows.append(res)
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(tables_dir / "survival_performance_test.csv", index=False)
    logger.info("Survival performance (test set):\n%s", perf_df.to_string(index=False))

    # ---- 2. cluster stability via subsampling + MOFA+k-means rerun -----------------------
    ecfg = cfg["evaluation"]
    ccfg = cfg["clustering"]
    kmeans_bundle = joblib.load(resolve(cfg["paths"]["checkpoints_dir"]) / "kmeans_model.joblib")
    k_star = kmeans_bundle["k"]
    ari_scores = []
    rng = np.random.RandomState(cfg["project"]["random_seed"])
    for i in range(ecfg["n_stability_subsamples"]):
        sub_ids = list(rng.choice(train_ids, size=int(ecfg["subsample_fraction"] * len(train_ids)), replace=False))
        logger.info("Stability run %d/%d: refitting MOFA+k-means on %d subsampled train patients",
                    i + 1, ecfg["n_stability_subsamples"], len(sub_ids))
        sub_labels = refit_mofa_cluster_subsample(proc_dir, sub_ids, cfg["mofa"]["views"], cfg["mofa"], k_star,
                                                   kmeans_seed=cfg["project"]["random_seed"] + i, n_init=ccfg["n_init_per_k"])
        ari = adjusted_rand_score(cl_train.loc[sub_ids], sub_labels.loc[sub_ids])
        ari_scores.append(ari)
        logger.info("Stability run %d: ARI = %.4f", i + 1, ari)

    stability_df = pd.DataFrame({"run": range(1, len(ari_scores) + 1), "ari": ari_scores})
    stability_df.to_csv(tables_dir / "cluster_stability_ari.csv", index=False)
    mean_ari = float(np.mean(ari_scores))
    logger.info("Mean ARI across %d subsamples: %.4f (acceptance threshold: %.2f) -> %s",
                len(ari_scores), mean_ari, ecfg["min_acceptable_mean_ari"],
                "PASS" if mean_ari >= ecfg["min_acceptable_mean_ari"] else "FAIL")

    # ---- 3. biological characterisation (train set) ---------------------------------------
    mrna_view = pd.read_parquet(proc_dir / VIEW_FILES["mrna_pathway"]).loc[train_ids]
    cna_view = pd.read_parquet(proc_dir / VIEW_FILES["cna_pathway"]).loc[train_ids]
    mut_view = pd.read_parquet(proc_dir / VIEW_FILES["mutation"]).loc[train_ids]
    mut_bin_cols = [c for c in mut_view.columns if c.startswith("MUT__")]

    logger.info("Describing per-cluster pathway activity (mRNA + CNA views; standardised mean differences) ...")
    pathway_desc = pd.concat([
        describe_features(mrna_view, cl_train, "mrna_pathway"),
        describe_features(cna_view, cl_train, "cna_pathway"),
    ], ignore_index=True)
    pathway_desc.to_csv(tables_dir / "cluster_biological_pathways.csv", index=False)

    logger.info("Describing recurrent mutation frequencies per cluster ...")
    mut_desc = describe_mutations(mut_view[mut_bin_cols], cl_train)
    mut_desc.to_csv(tables_dir / "cluster_biological_mutations.csv", index=False)

    # ---- merged table for downstream figure generation -------------------------------------
    all_labels = pd.concat([cl_train, cl_test]).rename("cluster")
    merged = clinical.join(all_labels, how="inner")
    merged.to_parquet(proc_dir / "clinical_with_clusters.parquet")

    logger.info("Stage 06 complete.")


if __name__ == "__main__":
    main()
