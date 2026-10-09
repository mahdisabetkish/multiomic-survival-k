#!/usr/bin/env python3
"""Stage 19: was the positive control's choice of k a boundary effect?

In the positive control (stage 16) the survival-guided rule chose k = 8 in most
replicates, and 8 was the largest value searched. A choice at the edge of the search
range is ambiguous: the rule may have wanted a larger k and been stopped, or 8 may be
the right answer.

There is a reason to expect the second. The simulated data contain two independent
groupings of the same observations, two nuisance groups and four signal groups, so the
latent space holds 2 x 4 = 8 distinct cells. Survival depends only on the signal group,
so every cell is homogeneous in hazard, and the eight-cell partition contains the
four-group survival partition as a coarsening. A rule that recovers the eight cells
has recovered all the survival-relevant structure there is.

This script reruns the positive control with k searched up to 15, and records the
agreement of each chosen partition with the signal grouping, the nuisance grouping and
the eight-cell product grouping. If the chosen k stays at 8 and agrees with the
product grouping, 8 was not a boundary effect. If it moves above 8, it was.

Output: outputs/tables/positive_control_k_range.csv
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from common import load_config, get_logger, resolve


def _stage16():
    path = Path(__file__).parent / "16_positive_control.py"
    spec = importlib.util.spec_from_file_location("stage16", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=15)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("19_positive_control_k_range", resolve(cfg["paths"]["logs_dir"]))
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    s16 = _stage16()
    ccfg = copy.deepcopy(cfg["clustering"])
    ccfg["k_max"] = args.k_max
    mcfg = cfg["mofa"]

    n_tr, n_te = 1302, 558
    rows = []
    for signal in (0.0, 0.8, 1.6):
        for rep in range(args.reps):
            # same seeds as stage 16, so the datasets are identical to the ones reported there
            rng = np.random.RandomState(5000 + int(signal * 100) + rep)
            views, t, d, g_sig, g_nuis = s16.simulate(n_tr + n_te, signal, nuisance_sep=9.0,
                                                      signal_sep=2.5, cens_scale=3.0, rng=rng)
            g_cell = g_nuis * s16.K_SIGNAL + g_sig
            sl, st = slice(0, n_tr), slice(n_tr, n_tr + n_te)
            Z_tr, Z_te = s16.fit_mofa_project({k: v[sl] for k, v in views.items()},
                                              {k: v[st] for k, v in views.items()},
                                              mcfg, seed_offset=200 + rep)
            k_surv, scores = s16.choose_k_survival(Z_tr, t[sl], d[sl], ccfg)
            k_sil, _ = s16.choose_k_silhouette(Z_tr, ccfg)
            ev = s16.evaluate(Z_tr, Z_te, k_surv, t[sl], d[sl], t[st], d[st],
                              g_nuis[sl], g_sig[st], ccfg)
            lab_te = KMeans(n_clusters=k_surv, n_init=ccfg["n_init_per_k"],
                            random_state=ccfg["random_seed"]).fit(Z_tr).predict(Z_te)
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            rows.append({"signal": signal, "rep": rep + 1, "k_max": args.k_max,
                         "k_survival": k_surv, "k_silhouette": k_sil,
                         "runner_up_k": ranked[1][0],
                         "winning_margin": ranked[0][1] - ranked[1][1],
                         "c_index_survival": ev["c_index"],
                         "ari_to_signal": adjusted_rand_score(g_sig[st], lab_te),
                         "ari_to_nuisance": adjusted_rand_score(g_nuis[st], lab_te),
                         "ari_to_product_cells": adjusted_rand_score(g_cell[st], lab_te)})
            logger.info("signal=%.1f rep %d: k_surv=%d (runner-up %d) k_sil=%d | C-index %.3f | "
                        "ARI to signal %.3f, nuisance %.3f, 8-cell product %.3f",
                        signal, rep + 1, k_surv, ranked[1][0], k_sil, ev["c_index"],
                        rows[-1]["ari_to_signal"], rows[-1]["ari_to_nuisance"],
                        rows[-1]["ari_to_product_cells"])

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "positive_control_k_range.csv", index=False)
    logger.info("")
    for sg, g in df.groupby("signal"):
        logger.info("signal %.1f: k chosen %s (search up to %d) | mean C-index %.3f | "
                    "mean ARI to product cells %.3f",
                    sg, sorted(g.k_survival.tolist()), args.k_max, g.c_index_survival.mean(),
                    g.ari_to_product_cells.mean())
    logger.info("Stage 19 complete.")


if __name__ == "__main__":
    main()
