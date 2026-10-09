#!/usr/bin/env python3
"""Stage 22: is the gain over silhouette selection a gain from the survival criterion, or from k > 2?

Across the repeated splits of stage 20 the survival-guided rule beat silhouette selection
in nine splits out of ten. Silhouette chose k = 2 on every split, while the survival rule
chose between 4 and 8. The comparison therefore confounds two things: the criterion used
to choose k, and the simple fact of having more than two clusters. If any k between 4 and
8 does as well on held-out data as the k the survival rule picks, the gain belongs to
"more clusters" and not to the criterion.

This script separates the two. On the same ten splits, with the same seeds, it rebuilds
the MOFA factors exactly as stage 20 does and then scores k-means at every fixed k from 2
to 15 on the held-out patients. Because the final k-means fit uses the same seed and
settings at every k, the survival-guided result on a split is one point on that split's
fixed-k curve, and the question becomes how good a point the rule picks. The rule is
compared with
    each fixed k in the search range 2-8,
    a k drawn uniformly from 2-8 (the per-split mean of the fixed-k curve),
    the best k on the held-out patients (an oracle, reported only as a ceiling).
It also records the selection score S(k) for k up to 15, so that the effect of widening
the search range on the real data can be read off directly.

The per-split ssGSEA scores and MOFA factors are cached under data/processed, which is
not under version control.

Outputs:
    outputs/tables/fixed_k_per_split.csv
    outputs/tables/fixed_k_summary.csv
    outputs/tables/fixed_k_comparisons.csv
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import load_config, get_logger, resolve


def _load(name):
    path = Path(__file__).parent / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def selection_scores(Z, dur, ev, ccfg, k_values, logrank_test):
    """S(k) for every k, with the seeds stage 05 and stage 20 use, so the argmax over 2-8 matches."""
    scores = {}
    for k in k_values:
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
    return scores


def split_features(j, clinical, mrna, mut_bin, cna_view, gene_sets, cfg, s20, s18, cache_dir, logger):
    """Factors for split j, built exactly as in stage 20; cached after the first run."""
    seed = 9000 + j
    tr_ids, te_ids = s20.stratified_split(clinical, cfg["split"]["test_fraction"], seed)
    f = cache_dir / f"split_{j + 1:02d}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["tr_ids"]) == list(tr_ids) and list(z["te_ids"]) == list(te_ids):
            logger.info("Split %2d: factors read from cache", j + 1)
            return tr_ids, te_ids, z["Z_tr"], z["Z_te"]

    var = mrna.loc[tr_ids].var(axis=0)
    keep = var[var >= np.percentile(var, cfg["preprocess"]["min_gene_variance_percentile"])].index
    mrna_view = s20.ssgsea_scores(mrna[keep], gene_sets, cfg)
    mrna_view.to_parquet(cache_dir / f"split_{j + 1:02d}_mrna_pathway.parquet")

    top_n = cfg["feature_aggregation"]["mutations"]["recurrent_gene_list_top_n"]
    rec = mut_bin.loc[tr_ids].sum(axis=0)
    rec = rec[rec >= cfg["preprocess"]["mutation_min_recurrence"]]
    top = rec.sort_values(ascending=False).head(top_n).index
    mut_view = mut_bin[top].copy()
    mut_view.columns = [f"MUT__{g}" for g in top]
    mut_view["TMB"] = mut_bin.sum(axis=1)

    order = cfg["mofa"]["views"]
    views = {"mrna_pathway": mrna_view, "cna_pathway": cna_view, "mutation": mut_view}
    scalers = {v: StandardScaler().fit(views[v].loc[tr_ids].values) for v in order}
    std_tr = {v: scalers[v].transform(views[v].loc[tr_ids].values) for v in order}
    std_te = {v: scalers[v].transform(views[v].loc[te_ids].values) for v in order}
    Z_tr, Z_te = s18.fit_mofa_project(std_tr, std_te, order, cfg["mofa"], seed_offset=500 + j)
    np.savez(f, Z_tr=Z_tr, Z_te=Z_te, tr_ids=np.array(tr_ids, dtype=object),
             te_ids=np.array(te_ids, dtype=object))
    del mrna_view, views, std_tr, std_te
    gc.collect()
    return tr_ids, te_ids, Z_tr, Z_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--k-max-wide", type=int, default=15)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("22_fixed_k_control", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    genesets_dir = resolve(cfg["paths"]["genesets_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    cache_dir = proc_dir / "repeated_splits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ccfg = cfg["clustering"]
    k_lo, k_hi = ccfg["k_min"], ccfg["k_max"]

    s03 = _load("03_feature_aggregation.py")
    s10 = _load("10_baseline_comparison.py")
    s18 = _load("18_geneset_seed_sensitivity.py")
    s20 = _load("20_repeated_outer_splits.py")
    from lifelines.statistics import multivariate_logrank_test

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    mrna = pd.read_parquet(proc_dir / "mrna_raw.parquet")
    cna = pd.read_parquet(proc_dir / "cna_raw.parquet")
    mut_bin = pd.read_parquet(proc_dir / "mutation_binary.parquet")
    gene_sets = s03.build_gene_sets(cfg, genesets_dir, logger)
    arm_map = s03.chrom_arm_map(genesets_dir / "hgnc_complete_set.txt")
    cna_view = pd.concat([s03.cna_arm_burden(cna, arm_map), s03.cna_pathway_scores(cna, gene_sets)], axis=1)

    ref = pd.read_csv(tables_dir / "repeated_splits_per_split.csv").set_index("split")
    k_all = list(range(k_lo, args.k_max_wide + 1))
    rows = []
    for j in range(args.splits):
        tr_ids, te_ids, Z_tr, Z_te = split_features(j, clinical, mrna, mut_bin, cna_view, gene_sets,
                                                    cfg, s20, s18, cache_dir, logger)
        dur_tr, ev_tr = clinical.loc[tr_ids, "OS_MONTHS_NUM"].values, clinical.loc[tr_ids, "OS_EVENT"].values
        dur_te, ev_te = clinical.loc[te_ids, "OS_MONTHS_NUM"].values, clinical.loc[te_ids, "OS_EVENT"].values

        S = selection_scores(Z_tr, dur_tr, ev_tr, ccfg, k_all, multivariate_logrank_test)
        for k in k_all:
            km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(Z_tr)
            c, p, _ = s10.cox_eval(km.labels_, km.predict(Z_te), dur_tr, ev_tr, dur_te, ev_te)
            rows.append({"split": j + 1, "k": k, "selection_score": S[k], "c_index_test": c,
                         "logrank_p_test": p})

        cur = pd.DataFrame([r for r in rows if r["split"] == j + 1]).set_index("k")
        k8 = max(range(k_lo, k_hi + 1), key=lambda k: S[k])
        k15 = max(k_all, key=lambda k: S[k])
        same = (k8 == ref.loc[j + 1, "k_survival"]
                and np.isclose(cur.loc[k8, "c_index_test"], ref.loc[j + 1, "c_survival"]))
        logger.info("Split %2d: chosen k %d in %d-%d (stage 20: %d, C-index %s), %d in %d-%d | "
                    "test C-index at chosen k %.3f, mean over %d-%d %.3f, best %.3f at k=%d",
                    j + 1, k8, k_lo, k_hi, ref.loc[j + 1, "k_survival"],
                    "reproduced" if same else "NOT reproduced", k15, k_lo, args.k_max_wide,
                    cur.loc[k8, "c_index_test"], k_lo, k_hi,
                    cur.loc[k_lo:k_hi, "c_index_test"].mean(), cur.loc[k_lo:k_hi, "c_index_test"].max(),
                    cur.loc[k_lo:k_hi, "c_index_test"].idxmax())
        del Z_tr, Z_te
        gc.collect()

    long = pd.DataFrame(rows)
    long.to_csv(tables_dir / "fixed_k_per_split.csv", index=False)

    wide = long.pivot(index="split", columns="k", values="c_index_test")
    score = long.pivot(index="split", columns="k", values="selection_score")
    rng_k = list(range(k_lo, k_hi + 1))
    k8 = score[rng_k].idxmax(axis=1)
    k15 = score[k_all].idxmax(axis=1)
    c8 = pd.Series([wide.loc[s, k8[s]] for s in wide.index], index=wide.index)
    c15 = pd.Series([wide.loc[s, k15[s]] for s in wide.index], index=wide.index)
    c_unif = wide[rng_k].mean(axis=1)
    c_best = wide[rng_k].max(axis=1)
    rank = pd.Series([int((wide.loc[s, rng_k] > c8[s]).sum()) + 1 for s in wide.index], index=wide.index)
    rho = pd.Series([stats.spearmanr(score.loc[s, rng_k], wide.loc[s, rng_k]).correlation
                     for s in wide.index], index=wide.index)

    summ = [{"quantity": f"fixed k = {k}", "mean_c_index": wide[k].mean(), "sd_c_index": wide[k].std(ddof=1)}
            for k in k_all]
    summ += [{"quantity": f"survival-guided k, searched {k_lo}-{k_hi}", "mean_c_index": c8.mean(),
              "sd_c_index": c8.std(ddof=1)},
             {"quantity": f"survival-guided k, searched {k_lo}-{args.k_max_wide}", "mean_c_index": c15.mean(),
              "sd_c_index": c15.std(ddof=1)},
             {"quantity": f"k uniform on {k_lo}-{k_hi}", "mean_c_index": c_unif.mean(),
              "sd_c_index": c_unif.std(ddof=1)},
             {"quantity": f"best k on held-out patients, {k_lo}-{k_hi} (oracle)", "mean_c_index": c_best.mean(),
              "sd_c_index": c_best.std(ddof=1)}]
    pd.DataFrame(summ).to_csv(tables_dir / "fixed_k_summary.csv", index=False)

    n_tr = len(clinical) - int(round(len(clinical) * cfg["split"]["test_fraction"]))
    n_te = len(clinical) - n_tr
    comp = []
    against = {f"fixed k = {k}": wide[k] for k in rng_k}
    against[f"k uniform on {k_lo}-{k_hi}"] = c_unif
    against[f"survival-guided k, searched {k_lo}-{args.k_max_wide}"] = c15
    for name, other in against.items():
        d = c8 - other
        _, p_c = s20.corrected_resampled_t(d, n_tr, n_te)
        comp.append({"comparison": f"survival-guided k ({k_lo}-{k_hi}) vs {name}",
                     "mean_difference": d.mean(), "sd_difference": d.std(ddof=1),
                     "splits_survival_better": int((d > 0).sum()), "splits_tied": int((d == 0).sum()),
                     "n_splits": len(d), "p_corrected_resampled_t": p_c})
    pd.DataFrame(comp).to_csv(tables_dir / "fixed_k_comparisons.csv", index=False)

    pd.DataFrame({"k_chosen_2_8": k8, "k_chosen_2_wide": k15, "c_chosen": c8, "c_uniform": c_unif,
                  "c_oracle": c_best, "rank_of_chosen_among_7": rank,
                  "spearman_score_vs_test_c": rho}).to_csv(tables_dir / "fixed_k_rule_per_split.csv")

    logger.info("")
    logger.info("Mean held-out C-index by fixed k: %s",
                ", ".join(f"{k}: {wide[k].mean():.3f}" for k in k_all))
    logger.info("Chosen k (%d-%d): %s; chosen k (%d-%d): %s", k_lo, k_hi, k8.tolist(), k_lo,
                args.k_max_wide, k15.tolist())
    logger.info("Survival-guided %.3f | uniform k %.3f | oracle %.3f | wide search %.3f",
                c8.mean(), c_unif.mean(), c_best.mean(), c15.mean())
    logger.info("Rank of the chosen k among the %d candidates (1 = best on held-out data): %s",
                len(rng_k), rank.tolist())
    logger.info("Spearman correlation of S(k) with held-out C-index over %d-%d, per split: %s (mean %.2f)",
                k_lo, k_hi, [round(x, 2) for x in rho], rho.mean())
    for c in comp:
        logger.info("%-70s diff %+.4f, better in %d/%d, corrected p %.3f", c["comparison"],
                    c["mean_difference"], c["splits_survival_better"], c["n_splits"],
                    c["p_corrected_resampled_t"])
    logger.info("Stage 22 complete.")


if __name__ == "__main__":
    main()
