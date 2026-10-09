#!/usr/bin/env python3
"""Stage 14: stability of the complete procedure, with k reselected each time.

The stability analysis in stage 06 holds k fixed at the value chosen on the full
training set, so it measures how reproducible the partition is given that k. It
does not measure how reproducible the procedure is, because the procedure also
chooses k, and that choice turns out to move.

This script repeats the whole discovery pipeline on subsamples of the training
set: refit the per-view scalers, refit MOFA, run the inner search to choose k,
and fit k-means with whatever k that search returns. Each resulting partition is
compared with the reference training partition on the patients they share, using
the adjusted Rand index. The ARI is well defined when the two partitions have
different numbers of groups, which is the case here more often than not.

Two comparisons are reported:
    conditional   k forced to the reference value (what stage 06 reports)
    unconditional k reselected inside each subsample (what the procedure does)

Output: outputs/tables/full_pipeline_stability.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point

from common import load_config, get_logger, resolve

VIEW_FILES = {
    "mrna_pathway": "view_mrna_pathway.parquet",
    "cna_pathway": "view_cna_pathway.parquet",
    "mutation": "view_mutation.parquet",
}


def fit_mofa(views_std, view_order, mcfg, seed_offset):
    n = views_std[view_order[0]].shape[0]
    ep = entry_point()
    ep.set_data_options(scale_views=False, center_groups=False)
    ep.set_data_matrix([[views_std[v]] for v in view_order],
                       likelihoods=["gaussian"] * len(view_order),
                       views_names=view_order, groups_names=["g"],
                       samples_names=[[f"s{i}" for i in range(n)]],
                       features_names=[[f"{v}_{i}" for i in range(views_std[v].shape[1])]
                                       for v in view_order])
    ep.set_model_options(factors=mcfg["n_factors_default"])
    ep.set_train_options(iter=mcfg["max_iterations"], convergence_mode=mcfg["convergence_mode"],
                         dropR2=mcfg["drop_factor_threshold"], seed=mcfg["seed"] + seed_offset,
                         quiet=True, verbose=False)
    ep.build()
    ep.run()
    return ep.model.nodes["Z"].getExpectation()


def inner_k_search(Z, durations, events, ccfg):
    scores = {}
    for k in range(ccfg["k_min"], ccfg["k_max"] + 1):
        rep = []
        for r in range(ccfg["n_inner_repeats"]):
            seed = ccfg["random_seed"] + 1000 * k + r
            tr, va = train_test_split(np.arange(len(Z)), test_size=ccfg["inner_val_fraction"],
                                      stratify=events, random_state=seed)
            km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=seed).fit(Z[tr])
            lab = km.predict(Z[va])
            if len(np.unique(lab)) < 2:
                rep.append(-np.inf)
                continue
            p = multivariate_logrank_test(durations[va], lab, events[va]).p_value
            rep.append(-np.log10(max(p, 1e-300)))
        scores[k] = float(np.mean(rep))
    return max(scores, key=scores.get), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("14_full_pipeline_stability", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ccfg, mcfg, ecfg = cfg["clustering"], cfg["mofa"], cfg["evaluation"]

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    reference = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    train_ids = reference.index
    k_ref = reference.nunique()
    views_raw = {v: pd.read_parquet(proc_dir / f).loc[train_ids] for v, f in VIEW_FILES.items()}
    view_order = mcfg["views"]
    dur = clinical.loc[train_ids, "OS_MONTHS_NUM"].values
    ev = clinical.loc[train_ids, "OS_EVENT"].values

    rng = np.random.RandomState(cfg["project"]["random_seed"])
    n_runs = ecfg["n_stability_subsamples"]
    size = int(ecfg["subsample_fraction"] * len(train_ids))
    rows = []

    for i in range(n_runs):
        pos = rng.choice(len(train_ids), size=size, replace=False)
        ids = train_ids[pos]
        scalers = {v: StandardScaler().fit(views_raw[v].values[pos]) for v in view_order}
        views_std = {v: scalers[v].transform(views_raw[v].values[pos]) for v in view_order}

        Z = fit_mofa(views_std, view_order, mcfg, seed_offset=100 + i)
        k_sel, scores = inner_k_search(Z, dur[pos], ev[pos], ccfg)

        lab_uncond = KMeans(n_clusters=k_sel, n_init=ccfg["n_init_per_k"],
                            random_state=ccfg["random_seed"]).fit_predict(Z)
        lab_cond = KMeans(n_clusters=k_ref, n_init=ccfg["n_init_per_k"],
                          random_state=ccfg["random_seed"]).fit_predict(Z)
        ref_sub = reference.loc[ids].values

        rows.append({
            "run": i + 1, "n_subsample": size, "n_factors": Z.shape[1],
            "k_selected": k_sel, "k_reference": k_ref,
            "ari_unconditional": adjusted_rand_score(ref_sub, lab_uncond),
            "ari_conditional": adjusted_rand_score(ref_sub, lab_cond),
            **{f"score_k{k}": round(v, 4) for k, v in scores.items()},
        })
        logger.info("Run %2d/%d: MOFA kept %2d factors, k selected = %d, "
                    "ARI unconditional = %.3f, ARI conditional on k=%d = %.3f",
                    i + 1, n_runs, Z.shape[1], k_sel, rows[-1]["ari_unconditional"],
                    k_ref, rows[-1]["ari_conditional"])

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "full_pipeline_stability.csv", index=False)
    logger.info("k selected across runs: %s", df["k_selected"].tolist())
    logger.info("Reference k was selected in %d of %d runs",
                int((df["k_selected"] == k_ref).sum()), n_runs)
    logger.info("Mean ARI, k reselected    : %.3f (range %.3f to %.3f)",
                df["ari_unconditional"].mean(), df["ari_unconditional"].min(),
                df["ari_unconditional"].max())
    logger.info("Mean ARI, k fixed at %d    : %.3f (range %.3f to %.3f)",
                k_ref, df["ari_conditional"].mean(), df["ari_conditional"].min(),
                df["ari_conditional"].max())
    logger.info("Stage 14 complete.")


if __name__ == "__main__":
    main()
