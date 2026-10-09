#!/usr/bin/env python3
"""Stage 17: is the instability in k a property of the criterion, or just too few inner repeats?

The selection score averages the log-rank statistic over R random inner splits, and
the paper uses R = 5. That invites an obvious objection: a mean of five noisy numbers
is itself noisy, so the fact that the selected k moves under resampling might say
nothing about the criterion and everything about the sample size of the average.

This script separates the two. For each of ten subsamples of the training set, the
scalers and MOFA are refitted and the selection is then run at R = 5, 15 and 50 inner
splits on the same latent space. If the selected k settles down as R grows, the
instability reported elsewhere is an artefact of averaging too few splits. If it does
not, the criterion itself does not identify k on this data, which is the stronger
claim and the one the paper makes.

Output: outputs/tables/inner_repeat_sensitivity.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point

from common import load_config, get_logger, resolve

VIEW_FILES = {
    "mrna_pathway": "view_mrna_pathway.parquet",
    "cna_pathway": "view_cna_pathway.parquet",
    "mutation": "view_mutation.parquet",
}
R_VALUES = (5, 15, 50)


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


def k_search(Z, dur, ev, ccfg, n_repeats, seed_base):
    """The selection of Section 3.6, with the number of inner repeats as an argument."""
    scores = {}
    for k in range(ccfg["k_min"], ccfg["k_max"] + 1):
        rep = []
        for r in range(n_repeats):
            seed = seed_base + 1000 * k + r
            tr, va = train_test_split(np.arange(len(Z)), test_size=ccfg["inner_val_fraction"],
                                      stratify=ev, random_state=seed)
            km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=seed).fit(Z[tr])
            lab = km.predict(Z[va])
            if len(np.unique(lab)) < 2:
                rep.append(-np.inf)
                continue
            p = multivariate_logrank_test(dur[va], lab, ev[va]).p_value
            rep.append(-np.log10(max(p, 1e-300)))
        scores[k] = (float(np.mean(rep)), float(np.std(rep)))
    best = max(scores, key=lambda k: scores[k][0])
    return best, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("17_inner_repeat_sensitivity", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ccfg, mcfg, ecfg = cfg["clustering"], cfg["mofa"], cfg["evaluation"]

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    reference = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    train_ids = reference.index
    views_raw = {v: pd.read_parquet(proc_dir / f).loc[train_ids] for v, f in VIEW_FILES.items()}
    view_order = mcfg["views"]
    dur = clinical.loc[train_ids, "OS_MONTHS_NUM"].values
    ev = clinical.loc[train_ids, "OS_EVENT"].values

    rng = np.random.RandomState(cfg["project"]["random_seed"])
    size = int(ecfg["subsample_fraction"] * len(train_ids))
    rows = []

    for i in range(args.runs):
        pos = rng.choice(len(train_ids), size=size, replace=False)
        scalers = {v: StandardScaler().fit(views_raw[v].values[pos]) for v in view_order}
        views_std = {v: scalers[v].transform(views_raw[v].values[pos]) for v in view_order}
        Z = fit_mofa(views_std, view_order, mcfg, seed_offset=300 + i)

        row = {"run": i + 1, "n_subsample": size, "n_factors": Z.shape[1]}
        for R in R_VALUES:
            k_sel, sc = k_search(Z, dur[pos], ev[pos], ccfg, R, ccfg["random_seed"] + 7919 * i)
            row[f"k_R{R}"] = k_sel
            row[f"score_R{R}"] = round(sc[k_sel][0], 4)
            # how far ahead of the runner-up the winner is, in units of its own spread
            ordered = sorted(sc.items(), key=lambda kv: -kv[1][0])
            gap = ordered[0][1][0] - ordered[1][1][0]
            row[f"gap_R{R}"] = round(gap, 4)
            row[f"sd_R{R}"] = round(sc[k_sel][1], 4)
        rows.append(row)
        logger.info("Run %2d/%d: k = %d (R=5), %d (R=15), %d (R=50)", i + 1, args.runs,
                    row["k_R5"], row["k_R15"], row["k_R50"])

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "inner_repeat_sensitivity.csv", index=False)

    logger.info("")
    for R in R_VALUES:
        ks = df[f"k_R{R}"].tolist()
        gaps = df[f"gap_R{R}"]
        sds = df[f"sd_R{R}"]
        logger.info("R=%2d: k selected %s | distinct values %d | mean winning margin %.3f, "
                    "mean within-k spread %.3f",
                    R, sorted(ks), df[f"k_R{R}"].nunique(), gaps.mean(), sds.mean())
    logger.info("Stage 17 complete.")


if __name__ == "__main__":
    main()
