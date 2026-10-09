#!/usr/bin/env python3
"""Stage 05: survival-aware k-means clustering in the MOFA latent space.

Outer loop over candidate k (number of clusters); for each k, k-means (with
many random initialisations via sklearn's n_init) is repeatedly fit on an
inner-train fold carved out of the training data and evaluated on a held-out
inner-validation fold. Two separation scores are computed on every inner
fold: the k-sample log-rank test (as -log10 p) and the concordance index of
a Cox model with cluster indicators as its only covariates. The k that
maximises the average of the pre-specified primary criterion
(config `selection_metric`) is selected; the other criterion is recorded
alongside it so that the choice of k can be checked against a criterion
that is not a p-value. Survival enters only here, in the choice of k: the
final partition is ordinary k-means on the factor scores with k fixed.
Test-set patients are assigned to the nearest of the final centroids
(already fit on train-only data), preserving the leakage-free split.

Outputs:
    outputs/tables/clustering_k_selection.csv
    data/processed/metabric/cluster_labels_train.parquet
    data/processed/metabric/cluster_labels_test.parquet
    outputs/checkpoints/kmeans_model.joblib
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from common import load_config, get_logger, resolve


METRICS = ("logrank_p", "cindex_diff")


def score_clustering(labels: np.ndarray, durations: np.ndarray, events: np.ndarray, metric: str) -> float:
    if len(np.unique(labels)) < 2:
        return -np.inf
    if metric == "logrank_p":
        p = multivariate_logrank_test(durations, labels, events).p_value
        return -np.log10(max(p, 1e-300))
    elif metric == "cindex_diff":
        # C-index on the inner-validation fold of a Cox model whose only covariates are the
        # cluster indicators (fit and scored on the same inner-validation patients, so this is a
        # measure of separation on that fold rather than a predictive estimate)
        df = pd.DataFrame({"duration": durations, "event": events})
        dummies = pd.get_dummies(labels, prefix="cluster", drop_first=True).astype(float)
        df = pd.concat([df, dummies], axis=1)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df, duration_col="duration", event_col="event")
        risk = cph.predict_partial_hazard(df)
        return concordance_index(durations, -risk, events)
    else:
        raise ValueError(f"Unknown selection_metric: {metric}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("05_survival_aware_clustering", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ckpt_dir = resolve(cfg["paths"]["checkpoints_dir"])

    ccfg = cfg["clustering"]
    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    Z_train = pd.read_parquet(proc_dir / "mofa_factors_train.parquet")
    Z_test = pd.read_parquet(proc_dir / "mofa_factors_test.parquet")

    train_clin = clinical.loc[Z_train.index]
    durations_all = train_clin["OS_MONTHS_NUM"].values
    events_all = train_clin["OS_EVENT"].values

    primary = ccfg["selection_metric"]
    if primary not in METRICS:
        raise ValueError(f"Unknown selection_metric: {primary}")
    results = []
    logger.info("Searching k in [%d, %d] with %d inner repeats, primary metric=%s (also recording %s)",
                ccfg["k_min"], ccfg["k_max"], ccfg["n_inner_repeats"], primary,
                [m for m in METRICS if m != primary])

    for k in range(ccfg["k_min"], ccfg["k_max"] + 1):
        repeat_scores = {m: [] for m in METRICS}
        for r in range(ccfg["n_inner_repeats"]):
            seed = ccfg["random_seed"] + 1000 * k + r
            inner_train_idx, inner_val_idx = train_test_split(
                np.arange(len(Z_train)), test_size=ccfg["inner_val_fraction"],
                stratify=events_all, random_state=seed,
            )
            km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=seed)
            km.fit(Z_train.values[inner_train_idx])
            val_labels = km.predict(Z_train.values[inner_val_idx])
            for m in METRICS:
                repeat_scores[m].append(score_clustering(
                    val_labels, durations_all[inner_val_idx], events_all[inner_val_idx], m))
        row = {"k": k}
        for m in METRICS:
            row[f"mean_{m}"] = float(np.mean(repeat_scores[m]))
            row[f"std_{m}"] = float(np.std(repeat_scores[m]))
        # backwards-compatible columns holding the primary criterion
        row["mean_score"], row["std_score"] = row[f"mean_{primary}"], row[f"std_{primary}"]
        results.append(row)
        logger.info("k=%d: " + ", ".join(f"mean {m} = {row[f'mean_{m}']:.4f} (+/- {row[f'std_{m}']:.4f})"
                                         for m in METRICS), k)

    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / "clustering_k_selection.csv", index=False)

    best_k = int(results_df.loc[results_df["mean_score"].idxmax(), "k"])
    logger.info("Selected k* = %d (best mean %s)", best_k, primary)
    for m in METRICS:
        if m != primary:
            alt_k = int(results_df.loc[results_df[f"mean_{m}"].idxmax(), "k"])
            logger.info("For reference, the %s criterion would have selected k = %d", m, alt_k)

    final_km = KMeans(n_clusters=best_k, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"])
    train_labels = final_km.fit_predict(Z_train.values)
    test_labels = final_km.predict(Z_test.values)

    train_out = pd.DataFrame({"cluster": train_labels}, index=Z_train.index)
    test_out = pd.DataFrame({"cluster": test_labels}, index=Z_test.index)
    train_out.to_parquet(proc_dir / "cluster_labels_train.parquet")
    test_out.to_parquet(proc_dir / "cluster_labels_test.parquet")

    joblib.dump({"model": final_km, "k": best_k, "factor_cols": list(Z_train.columns)},
                ckpt_dir / "kmeans_model.joblib")

    logger.info("Train cluster sizes:\n%s", train_out["cluster"].value_counts().sort_index().to_string())
    logger.info("Test cluster sizes:\n%s", test_out["cluster"].value_counts().sort_index().to_string())
    logger.info("Stage 05 complete.")


if __name__ == "__main__":
    main()
