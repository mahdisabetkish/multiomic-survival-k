#!/usr/bin/env python3
"""Stage 21: is the selection score biased towards larger k when there is nothing to find?

The rule for choosing k maximises -log10 of a k-sample log-rank p-value computed on an
inner-validation fold of about 326 patients. If the test is valid, then with no
association between groups and survival the p-value is uniform for every k: the mean of
-log10 p is 1/ln(10) = 0.434 and five per cent of tests reject at the 0.05 level,
whatever the number of groups. Any systematic increase with k would bias the rule towards
larger k regardless of the data.

The log-rank statistic is referred to a chi-square distribution, and that approximation
can deteriorate when a fold of fixed size is divided into many small groups. This script
checks the calibration directly: survival times independent of the labels, censoring at
roughly the level in the data, and labels forming k equal-sized random groups.

Equal-sized random groups are the favourable case. Groups produced by k-means are unequal
and can be very small, so any miscalibration found here is if anything understated.

Output: outputs/tables/logrank_null_calibration.csv
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test

from common import load_config, get_logger, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--n", type=int, default=326)
    args = ap.parse_args()
    cfg = load_config(args.config)
    logger = get_logger("21_logrank_null_calibration", resolve(cfg["paths"]["logs_dir"]))
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    warnings.filterwarnings("ignore")

    rng = np.random.RandomState(1)
    rows = []
    for k in (2, 4, 6, 8, 10, 12, 15):
        lp, rej = [], 0
        for _ in range(args.reps):
            t = rng.exponential(1.0, args.n)
            c = rng.exponential(3.0, args.n)
            T, D = np.minimum(t, c), (t <= c).astype(int)
            lab = rng.permutation(np.arange(args.n) % k)
            p = multivariate_logrank_test(T, lab, D).p_value
            lp.append(-np.log10(max(p, 1e-300)))
            rej += p < 0.05
        lp = np.array(lp)
        rows.append({"k": k, "group_size": args.n // k, "reps": args.reps,
                     "mean_neg_log10_p": lp.mean(), "se_mean": lp.std(ddof=1) / np.sqrt(args.reps),
                     "rejection_rate_005": rej / args.reps})
        logger.info("k=%2d (groups of %3d): mean -log10 p %.3f (SE %.3f), P(p<0.05) %.3f",
                    k, args.n // k, rows[-1]["mean_neg_log10_p"], rows[-1]["se_mean"],
                    rows[-1]["rejection_rate_005"])
    pd.DataFrame(rows).to_csv(tables_dir / "logrank_null_calibration.csv", index=False)
    logger.info("Nominal values for a valid test: mean -log10 p 0.434, P(p<0.05) 0.050")
    logger.info("Stage 21 complete.")


if __name__ == "__main__":
    main()
