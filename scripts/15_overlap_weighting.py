#!/usr/bin/env python3
"""Stage 15: does a different weighting scheme achieve covariate balance?

Stage 07 uses stabilised inverse-probability-of-treatment weights and reports that
they leave tumour stage, hormone-therapy co-administration and nodal status well
above the conventional balance threshold. Proceeding to a weighted outcome model
when the weighting has demonstrably failed is hard to defend, so this script asks
whether a different estimand and weighting scheme does better on the same data.

Three schemes are compared on exactly the same cohort and confounder matrix:

    IPTW        stabilised inverse-probability weights, trimmed (stage 07)
    Overlap     weights w = 1 - e for the treated and w = e for the untreated
    Matching    1:1 nearest-neighbour on the propensity score with a calliper

Overlap weights (Li, Morgan and Zaslavsky) are of interest here because they
minimise the variance of the weighted difference and, for a propensity model fitted
by logistic regression, give exact mean balance on every covariate in that model.
The price is that the estimand changes: it is the effect in the subpopulation with
genuine clinical equipoise rather than in the whole cohort, which for 82 treated
patients among 980 is arguably the only population the data can speak about.

Output: outputs/tables/causal_weighting_comparison.csv
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.linear_model import LogisticRegression

from common import load_config, get_logger, resolve


def _load_stage07():
    path = Path(__file__).parent / "07_causal_treatment_effects.py"
    spec = importlib.util.spec_from_file_location("stage07", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def balance_table(confounders, treatment, weights, smd_fn):
    rows = []
    for col in confounders.columns:
        x = confounders[col].values.astype(float)
        rows.append({"covariate": col, "smd": smd_fn(x, treatment, weights)})
    return pd.DataFrame(rows)


def nn_match(propensity, treatment, calliper_sd=0.2, rng=None):
    """1:1 nearest-neighbour matching without replacement on the logit of the score."""
    logit = np.log(np.clip(propensity, 1e-6, 1 - 1e-6) / (1 - np.clip(propensity, 1e-6, 1 - 1e-6)))
    calliper = calliper_sd * logit.std(ddof=1)
    treated = np.where(treatment == 1)[0]
    control = list(np.where(treatment == 0)[0])
    weights = np.zeros(len(treatment))
    order = treated[np.argsort(-propensity[treated])]      # hardest to match first
    for t in order:
        if not control:
            break
        d = np.abs(logit[control] - logit[t])
        j = int(np.argmin(d))
        if d[j] <= calliper:
            weights[t] = 1.0
            weights[control[j]] = 1.0
            control.pop(j)
    return weights


def fit_weighted_cox(confounders, treatment, cluster, durations, events, weights, logger, label):
    keep = weights > 0
    df = confounders.loc[keep].copy()
    df["treatment"] = treatment.values[keep]
    df["cluster"] = cluster.values[keep]
    df["duration"], df["event"], df["weight"] = durations[keep], events[keep], weights[keep]
    terms = " + ".join(confounders.columns)
    try:
        full = CoxPHFitter(penalizer=0.1)
        full.fit(df, duration_col="duration", event_col="event", weights_col="weight",
                 formula=f"treatment * C(cluster) + {terms}", robust=True)
        red = CoxPHFitter(penalizer=0.1)
        red.fit(df, duration_col="duration", event_col="event", weights_col="weight",
                formula=f"treatment + C(cluster) + {terms}", robust=True)
        lr = 2 * (full.log_likelihood_ - red.log_likelihood_)
        ddf = full.params_.shape[0] - red.params_.shape[0]
        p = stats.chi2.sf(lr, ddf) if ddf > 0 else np.nan
        return {"n_used": int(keep.sum()), "n_treated_used": int(((treatment == 1) & keep).sum()),
                "lr_statistic": lr, "df": ddf, "lrt_p": p}
    except Exception as e:
        logger.warning("Cox fit failed for %s (%s)", label, e)
        return {"n_used": int(keep.sum()), "n_treated_used": int(((treatment == 1) & keep).sum()),
                "lr_statistic": np.nan, "df": np.nan, "lrt_p": np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("15_overlap_weighting", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    stage07 = _load_stage07()
    ccfg = cfg["causal"]

    clinical = pd.read_parquet(proc_dir / "clinical_with_clusters.parquet")
    covariates = pd.read_parquet(proc_dir / "clinical_covariates.parquet")
    labels = pd.read_parquet(proc_dir / "cluster_labels_train_oof.parquet")["cluster"]
    train_ids = labels.index

    cohort = clinical.loc[train_ids].copy()
    fcol, fval = next(iter(ccfg["cohort_filter"].items()))
    cohort = cohort[cohort[fcol] == fval]
    cohort = cohort.dropna(subset=[ccfg["treatment_col"], "OS_MONTHS_NUM", "OS_EVENT"])
    cohort = cohort[cohort[ccfg["treatment_col"]] != ""]

    treatment = (cohort[ccfg["treatment_col"]] == ccfg["treatment_positive_value"]).astype(int)
    durations, events = cohort["OS_MONTHS_NUM"].values, cohort["OS_EVENT"].values
    cluster = labels.loc[cohort.index].astype(int)
    confounders = stage07.build_confounder_matrix(covariates, cohort.index, ccfg["confounders"])
    cluster_dummies = pd.get_dummies(cluster, prefix="cluster", drop_first=True).astype(float)
    X_ps = pd.concat([confounders, cluster_dummies], axis=1)

    ps = LogisticRegression(max_iter=2000, C=1.0).fit(X_ps.values, treatment.values)
    e = np.clip(ps.predict_proba(X_ps.values)[:, 1], 1e-3, 1 - 1e-3)
    logger.info("Cohort: n=%d, treated=%d. Propensity range [%.3f, %.3f]",
                len(cohort), int(treatment.sum()), e.min(), e.max())

    t = treatment.values
    schemes = {}

    p_t = t.mean()
    sw = np.where(t == 1, p_t / e, (1 - p_t) / (1 - e))
    lo, hi = np.percentile(sw, ccfg["iptw_trim_percentile"])
    schemes["IPTW (stabilised, trimmed)"] = np.clip(sw, lo, hi)
    schemes["Overlap"] = np.where(t == 1, 1 - e, e)
    schemes["1:1 matched (0.2 SD calliper)"] = nn_match(e, t)

    smd = stage07.standardized_mean_diff
    rows, bal_frames = [], []
    unweighted = balance_table(confounders, t, None, smd).rename(columns={"smd": "Unweighted"})
    bal_frames.append(unweighted.set_index("covariate"))

    for name, w in schemes.items():
        b = balance_table(confounders, t, w, smd)
        bal_frames.append(b.set_index("covariate").rename(columns={"smd": name}))
        fit = fit_weighted_cox(confounders, treatment, cluster, durations, events, w, logger, name)
        rows.append({"scheme": name,
                     "max_abs_smd": b["smd"].abs().max(),
                     "mean_abs_smd": b["smd"].abs().mean(),
                     "n_covariates_above_0.1": int((b["smd"].abs() > 0.1).sum()),
                     "n_covariates": len(b), **fit})
        logger.info("%-32s max|SMD|=%.3f  mean|SMD|=%.3f  above 0.1: %d/%d  n=%d (treated %d)  LRT p=%s",
                    name, rows[-1]["max_abs_smd"], rows[-1]["mean_abs_smd"],
                    rows[-1]["n_covariates_above_0.1"], rows[-1]["n_covariates"],
                    fit["n_used"], fit["n_treated_used"],
                    "n/a" if np.isnan(fit["lrt_p"]) else f"{fit['lrt_p']:.3f}")

    u = unweighted.set_index("covariate")["Unweighted"]
    logger.info("Unweighted                       max|SMD|=%.3f  above 0.1: %d/%d",
                u.abs().max(), int((u.abs() > 0.1).sum()), len(u))

    pd.DataFrame(rows).to_csv(tables_dir / "causal_weighting_comparison.csv", index=False)
    pd.concat(bal_frames, axis=1).round(4).to_csv(tables_dir / "causal_weighting_balance.csv")
    logger.info("Stage 15 complete.")


if __name__ == "__main__":
    main()
