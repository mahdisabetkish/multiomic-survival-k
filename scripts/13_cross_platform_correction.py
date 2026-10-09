#!/usr/bin/env python3
"""Stage 13: quantify and attempt to correct the METABRIC/TCGA cross-platform
batch effect identified in stage 08 (microarray vs RNA-seq).

1. Quantifies the raw batch effect: per-feature standardized mean difference
   (Cohen's d) between METABRIC-train and TCGA ssGSEA/CNA-pathway scores.
2. Applies ComBat (Johnson, Li & Rabinovic 2007; `pycombat` implementation)
   to jointly harmonize METABRIC-train + TCGA pathway features per view,
   treating cohort as the batch variable, and re-quantifies the residual
   batch effect.
3. Refits MOFA and the survival-aware clustering on the harmonized
   METABRIC-train data (mutation view unchanged -- TCGA never had it), then
   re-projects and re-clusters TCGA under the harmonized model, and
   re-evaluates prognostic separation and cluster-proportion consistency,
   for direct comparison against stage 08's uncorrected numbers.

Outputs (outputs/tables/):
    cross_platform_batch_effect_magnitude.csv
    cross_platform_correction_comparison.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from pycombat import Combat
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point

from common import load_config, get_logger, resolve


def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = a.shape[0], b.shape[0]
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    pooled_sd = np.where(pooled_sd < 1e-8, 1e-8, pooled_sd)
    return (a.mean(axis=0) - b.mean(axis=0)) / pooled_sd


def survival_aware_k_search(Z_train, dur_tr, ev_tr, ccfg, logger):
    results = []
    for k in range(ccfg["k_min"], ccfg["k_max"] + 1):
        repeat_scores = []
        for r in range(ccfg["n_inner_repeats"]):
            seed = ccfg["random_seed"] + 1000 * k + r
            inner_train_idx, inner_val_idx = train_test_split(
                np.arange(len(Z_train)), test_size=ccfg["inner_val_fraction"], stratify=ev_tr, random_state=seed)
            km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=seed).fit(Z_train[inner_train_idx])
            val_labels = km.predict(Z_train[inner_val_idx])
            if len(np.unique(val_labels)) < 2:
                repeat_scores.append(-np.inf)
                continue
            p = multivariate_logrank_test(dur_tr[inner_val_idx], val_labels, ev_tr[inner_val_idx]).p_value
            repeat_scores.append(-np.log10(max(p, 1e-300)))
        results.append((k, float(np.mean(repeat_scores))))
    best_k = max(results, key=lambda x: x[1])[0]
    logger.info("Harmonized k search: %s -> k*=%d", results, best_k)
    return KMeans(n_clusters=best_k, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(Z_train), best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("13_cross_platform_correction", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    proc_tcga_dir = resolve(cfg["paths"]["processed_tcga_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ccfg, mcfg = cfg["clustering"], cfg["mofa"]

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    train_ids = clinical.index[clinical["split"] == "train"]
    dur_tr, ev_tr = clinical.loc[train_ids, "OS_MONTHS_NUM"].values, clinical.loc[train_ids, "OS_EVENT"].values

    tcga_clin = pd.read_parquet(proc_tcga_dir / "clinical_with_clusters.parquet")
    tcga_ids = tcga_clin.index
    dur_tcga, ev_tcga = tcga_clin["OS_MONTHS_NUM"].values, tcga_clin["OS_EVENT"].values

    view_cols = {}
    harmonized = {}
    batch_effect_rows = []
    for view in ["view_mrna_pathway", "view_cna_pathway"]:
        mb = pd.read_parquet(proc_dir / f"{view}.parquet").loc[train_ids]
        tc = pd.read_parquet(proc_tcga_dir / f"{view}.parquet").loc[tcga_ids]
        shared_cols = [c for c in mb.columns if c in tc.columns]
        mb, tc = mb[shared_cols], tc[shared_cols]
        view_cols[view] = shared_cols

        d_raw = cohens_d(mb.values, tc.values)

        Y = np.vstack([mb.values, tc.values])
        b = np.array([0] * len(mb) + [1] * len(tc))
        Y_harm = Combat().fit_transform(Y, b)
        mb_h, tc_h = Y_harm[:len(mb)], Y_harm[len(mb):]
        d_harm = cohens_d(mb_h, tc_h)

        batch_effect_rows.append({
            "view": view, "n_features": len(shared_cols),
            "mean_abs_cohens_d_before": np.abs(d_raw).mean(), "median_abs_cohens_d_before": np.median(np.abs(d_raw)),
            "pct_features_|d|>0.5_before": (np.abs(d_raw) > 0.5).mean() * 100,
            "mean_abs_cohens_d_after": np.abs(d_harm).mean(), "median_abs_cohens_d_after": np.median(np.abs(d_harm)),
            "pct_features_|d|>0.5_after": (np.abs(d_harm) > 0.5).mean() * 100,
        })
        harmonized[view] = (pd.DataFrame(mb_h, index=mb.index, columns=shared_cols),
                             pd.DataFrame(tc_h, index=tc.index, columns=shared_cols))
        logger.info("%s: mean|d| %.3f -> %.3f after ComBat (%d features)",
                    view, np.abs(d_raw).mean(), np.abs(d_harm).mean(), len(shared_cols))

    pd.DataFrame(batch_effect_rows).to_csv(tables_dir / "cross_platform_batch_effect_magnitude.csv", index=False)

    # ---- refit MOFA + survival-aware clustering on harmonized METABRIC-train data -----------
    mut_view = pd.read_parquet(proc_dir / "view_mutation.parquet").loc[train_ids]
    view_order = ["mrna_pathway", "cna_pathway", "mutation"]
    views_train_raw = {"mrna_pathway": harmonized["view_mrna_pathway"][0],
                        "cna_pathway": harmonized["view_cna_pathway"][0], "mutation": mut_view}
    views_tcga_raw = {"mrna_pathway": harmonized["view_mrna_pathway"][1],
                       "cna_pathway": harmonized["view_cna_pathway"][1]}

    scalers = {v: StandardScaler().fit(views_train_raw[v].values) for v in view_order}
    views_train_std = {v: scalers[v].transform(views_train_raw[v].values) for v in view_order}
    views_tcga_std = {v: scalers[v].transform(views_tcga_raw[v].values) for v in ["mrna_pathway", "cna_pathway"]}

    feat_names = [[f"{v}_{i}" for i in range(views_train_std[v].shape[1])] for v in view_order]
    data = [[views_train_std[v]] for v in view_order]
    ep = entry_point()
    ep.set_data_options(scale_views=False, center_groups=False)
    ep.set_data_matrix(data, likelihoods=["gaussian"] * len(view_order), views_names=view_order,
                        groups_names=["g"], samples_names=[list(train_ids)], features_names=feat_names)
    ep.set_model_options(factors=mcfg["n_factors_default"])
    ep.set_train_options(iter=mcfg["max_iterations"], convergence_mode=mcfg["convergence_mode"],
                          dropR2=mcfg["drop_factor_threshold"], seed=mcfg["seed"] + 2, quiet=True, verbose=False)
    ep.build()
    ep.run()
    Z_train = ep.model.nodes["Z"].getExpectation()
    n_factors = Z_train.shape[1]
    W = {v: ep.model.nodes["W"].nodes[m].getExpectation() for m, v in enumerate(view_order)}
    tau = {v: ep.model.nodes["Tau"].nodes[m].getExpectation().mean(axis=0) for m, v in enumerate(view_order)}
    logger.info("Harmonized-model MOFA retained %d factors", n_factors)

    final_km, k_star = survival_aware_k_search(Z_train, dur_tr, ev_tr, ccfg, logger)
    train_labels = final_km.labels_

    # project TCGA (mrna+cna views only, as in stage 08) into the harmonized MOFA space
    precision = np.eye(n_factors)
    rhs = np.zeros((len(tcga_ids), n_factors))
    for v in ["mrna_pathway", "cna_pathway"]:
        precision += W[v].T @ (W[v] * tau[v][:, None])
        rhs += views_tcga_std[v] @ (W[v] * tau[v][:, None])
    Z_tcga = rhs @ np.linalg.inv(precision).T
    tcga_labels = final_km.predict(Z_tcga)

    logrank_p = multivariate_logrank_test(dur_tcga, tcga_labels, ev_tcga).p_value
    dummies = pd.get_dummies(tcga_labels, prefix="c", drop_first=True).astype(float)
    df_cox = dummies.copy()
    df_cox["duration"], df_cox["event"] = dur_tcga, ev_tcga
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(df_cox, duration_col="duration", event_col="event")
        risk = cph.predict_partial_hazard(dummies).values.ravel()
        c_index = concordance_index(dur_tcga, -risk, ev_tcga)
    except Exception:
        c_index = np.nan

    metabric_props = pd.Series(train_labels).value_counts(normalize=True).sort_index()
    tcga_props = pd.Series(tcga_labels).value_counts(normalize=True).sort_index()

    before = pd.read_csv(tables_dir / "tcga_survival_validation.csv").iloc[0]
    comparison = pd.DataFrame([
        {"metric": "k", "before_combat": before["n_clusters"], "after_combat": k_star},
        {"metric": "logrank_p", "before_combat": before["logrank_p"], "after_combat": logrank_p},
        {"metric": "c_index", "before_combat": before["c_index_cluster_only"], "after_combat": c_index},
        {"metric": "max_cluster_proportion_shift_vs_train",
         "before_combat": pd.read_csv(tables_dir / "tcga_cluster_proportions_comparison.csv")
             .assign(shift=lambda d: (d["metabric_train_proportion"] - d["tcga_proportion"]).abs())["shift"].max(),
         "after_combat": (metabric_props.reindex(tcga_props.index.union(metabric_props.index), fill_value=0)
                           - tcga_props.reindex(tcga_props.index.union(metabric_props.index), fill_value=0)).abs().max()},
    ])
    comparison.to_csv(tables_dir / "cross_platform_correction_comparison.csv", index=False)
    logger.info("Cross-platform correction comparison:\n%s", comparison.to_string(index=False))
    logger.info("Stage 13 complete.")


if __name__ == "__main__":
    main()
