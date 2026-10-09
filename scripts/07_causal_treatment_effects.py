#!/usr/bin/env python3
"""Stage 07: causal analysis of treatment effect heterogeneity.

Contrast: adjuvant chemotherapy in ER-positive patients (config `causal`
section), analysed on the TRAINING split only (per the study protocol).

Cluster labels used here are obtained OUT OF FOLD. The training-set labels
from stage 05 were produced by a procedure that chose k using training
survival, so fitting a survival model with a treatment x cluster interaction
on those same patients conditions on a partition that was selected, in
part, on the outcome being modelled. To remove this, the training set is
split into F folds; for each fold, k is chosen by the same inner search as
stage 05 and the k-means centroids are fitted using only the other F-1
folds, and the held-out fold is assigned to those centroids. Every patient's
label is therefore obtained without their own survival time. Labels are
matched across folds to the stage 05 reference centroids by Hungarian
assignment on centroid distances (a relabelling only; when a fold selects a
different k, its clusters are mapped to the nearest reference centroid and
this is recorded). The same analysis is repeated with the in-sample stage 05
labels and written with the suffix `_insample_labels` for comparison.

  1. Propensity scores from logistic regression on clinical confounders +
     cluster membership.
  2. Stabilised inverse-probability-of-treatment weights (IPTW), trimmed at
     the 1st/99th percentile.
  3. IPTW-weighted Cox model with a treatment x cluster interaction to obtain
     cluster-specific hazard ratios; a likelihood-ratio test against the
     no-interaction model formally tests for effect modification.
  4. A causal forest (EconML CausalForestDML) estimates heterogeneous
     treatment effects non-parametrically. Since CausalForestDML expects an
     uncensored continuous/binary outcome and provides no built-in survival
     support, the outcome is a restricted-mean-survival-time (RMST) jackknife
     pseudo-observation at a fixed horizon (Andersen et al. 2004) -- a
     standard way to turn a censored survival outcome into a per-patient
     continuous value usable by any conditional-mean causal estimator.
  5. E-values quantify sensitivity of the cluster-specific hazard ratios to
     unmeasured confounding.

Outputs (outputs/tables/):
    causal_oof_fold_summary.csv       (k chosen per fold; agreement with in-sample labels)
    causal_propensity_diagnostics.csv
    causal_propensity_balance.csv
    causal_cox_forest_plot.csv        (cluster-specific HRs, CIs, E-values)
    causal_forest_cate_by_cluster.csv (causal-forest CATEs, CIs)
    causal_effect_modification_lrt.csv
    ... and the same five files with suffix `_insample_labels`
data/processed/metabric/cluster_labels_train_oof.parquet
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import restricted_mean_survival_time
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from econml.dml import CausalForestDML

from common import load_config, get_logger, resolve


def evalue(estimate: float) -> float:
    """E-value for a hazard-ratio-like estimate (Vanderweele & Ding 2017)."""
    rr = estimate
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def jackknife_rmst_pseudo_obs(durations: np.ndarray, events: np.ndarray, horizon: float) -> np.ndarray:
    n = len(durations)
    kmf_full = KaplanMeierFitter().fit(durations, events)
    rmst_full = restricted_mean_survival_time(kmf_full, t=horizon)

    pseudo = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        kmf_i = KaplanMeierFitter().fit(durations[mask], events[mask])
        rmst_i = restricted_mean_survival_time(kmf_i, t=horizon)
        pseudo[i] = n * rmst_full - (n - 1) * rmst_i
    return pseudo


def standardized_mean_diff(x: np.ndarray, treatment: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Standardized mean difference between treated and control, optionally weighted (Austin 2009)."""
    t_mask, c_mask = treatment == 1, treatment == 0
    if weights is None:
        mt, mc = x[t_mask].mean(), x[c_mask].mean()
        vt, vc = x[t_mask].var(ddof=1), x[c_mask].var(ddof=1)
    else:
        wt, wc = weights[t_mask], weights[c_mask]
        mt = np.average(x[t_mask], weights=wt)
        mc = np.average(x[c_mask], weights=wc)
        vt = np.average((x[t_mask] - mt) ** 2, weights=wt)
        vc = np.average((x[c_mask] - mc) ** 2, weights=wc)
    pooled_sd = np.sqrt((vt + vc) / 2)
    return (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0


def build_confounder_matrix(covariates: pd.DataFrame, ids, confounder_cols) -> pd.DataFrame:
    numeric = ["AGE_AT_DIAGNOSIS", "TUMOR_SIZE", "TUMOR_STAGE", "LYMPH_NODES_EXAMINED_POSITIVE", "GRADE"]
    num_cols = [c for c in confounder_cols if c in numeric]
    cat_cols = [c for c in confounder_cols if c not in numeric]

    num = covariates.loc[ids, num_cols].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    num_imp = pd.DataFrame(imputer.fit_transform(num), index=ids, columns=num_cols)

    cat = covariates.loc[ids, cat_cols].fillna("Unknown").replace("", "Unknown")
    cat_dummies = pd.get_dummies(cat, drop_first=True)

    return pd.concat([num_imp, cat_dummies], axis=1).astype(float)


def survival_aware_k_search(Z, durations, events, ccfg):
    """Same inner search as stage 05 (log-rank criterion), on the array Z passed in."""
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


def match_to_reference(fold_centroids: np.ndarray, ref_centroids: np.ndarray):
    """Map fold cluster indices to reference cluster indices.

    Equal k: one-to-one Hungarian assignment on Euclidean centroid distances (a pure relabelling).
    Different k: each fold centroid goes to its nearest reference centroid (many-to-one), which
    merges fold clusters; the caller records that this happened.
    """
    d = np.linalg.norm(fold_centroids[:, None, :] - ref_centroids[None, :, :], axis=2)
    if fold_centroids.shape[0] == ref_centroids.shape[0]:
        rows, cols = linear_sum_assignment(d)
        return dict(zip(rows, cols)), False
    return {i: int(d[i].argmin()) for i in range(fold_centroids.shape[0])}, True


def out_of_fold_labels(Z_train: pd.DataFrame, durations, events, ccfg, n_folds, seed, ref_centroids, logger):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = pd.Series(-1, index=Z_train.index, dtype=int)
    fold_rows = []
    Z = Z_train.values
    for f, (fit_idx, held_idx) in enumerate(skf.split(Z, events), 1):
        k_f, scores = survival_aware_k_search(Z[fit_idx], durations[fit_idx], events[fit_idx], ccfg)
        km = KMeans(n_clusters=k_f, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(Z[fit_idx])
        raw = km.predict(Z[held_idx])
        mapping, merged = match_to_reference(km.cluster_centers_, ref_centroids)
        labels.iloc[held_idx] = [mapping[r] for r in raw]
        fold_rows.append({"fold": f, "n_fit": len(fit_idx), "n_held_out": len(held_idx),
                          "k_selected": k_f, "clusters_merged_on_matching": merged,
                          **{f"score_k{k}": v for k, v in scores.items()}})
        logger.info("OOF fold %d/%d: k*=%d (scores %s)%s", f, n_folds, k_f,
                    {k: round(v, 3) for k, v in scores.items()},
                    "; k differs from reference, fold clusters mapped to nearest reference centroid" if merged else "")
    assert (labels >= 0).all()
    return labels, pd.DataFrame(fold_rows)


def run_causal_analysis(cohort: pd.DataFrame, covariates: pd.DataFrame, cluster: pd.Series, cfg: dict,
                        tables_dir, logger, suffix: str = ""):
    ccfg = cfg["causal"]
    treatment = (cohort[ccfg["treatment_col"]] == ccfg["treatment_positive_value"]).astype(int)
    durations, events = cohort["OS_MONTHS_NUM"].values, cohort["OS_EVENT"].values
    cluster = cluster.loc[cohort.index].astype(int)
    tag = suffix if suffix else " (out-of-fold labels)"
    logger.info("=== Causal analysis%s: n=%d, treated=%d, clusters=%s ===", tag, len(cohort),
                int(treatment.sum()), cluster.value_counts().sort_index().to_dict())

    # ---- 1-2. propensity scores + stabilised IPTW ------------------------------------------
    confounders = build_confounder_matrix(covariates, cohort.index, ccfg["confounders"])
    cluster_dummies = pd.get_dummies(cluster, prefix="cluster", drop_first=True).astype(float)
    X_ps = pd.concat([confounders, cluster_dummies], axis=1)

    ps_model = LogisticRegression(max_iter=2000, C=1.0)
    ps_model.fit(X_ps.values, treatment.values)
    propensity = ps_model.predict_proba(X_ps.values)[:, 1]
    propensity = np.clip(propensity, 1e-3, 1 - 1e-3)

    p_treated = treatment.mean()
    sw = np.where(treatment == 1, p_treated / propensity, (1 - p_treated) / (1 - propensity))
    lo, hi = np.percentile(sw, ccfg["iptw_trim_percentile"])
    sw_trimmed = np.clip(sw, lo, hi)

    diag = pd.DataFrame({
        "patient_id": cohort.index, "cluster": cluster.values, "treatment": treatment.values,
        "propensity": propensity, "stabilized_weight": sw, "stabilized_weight_trimmed": sw_trimmed,
    })
    diag.to_csv(tables_dir / f"causal_propensity_diagnostics{suffix}.csv", index=False)
    logger.info("Propensity score range: [%.3f, %.3f]; weight range after trim: [%.3f, %.3f]",
                propensity.min(), propensity.max(), sw_trimmed.min(), sw_trimmed.max())

    # ---- covariate balance before/after IPTW weighting -------------------------------------
    t_arr = treatment.values
    balance_rows = []
    for col in confounders.columns:
        x = confounders[col].values.astype(float)
        balance_rows.append({
            "covariate": col,
            "smd_before_weighting": standardized_mean_diff(x, t_arr),
            "smd_after_weighting": standardized_mean_diff(x, t_arr, weights=sw_trimmed),
        })
    balance = pd.DataFrame(balance_rows)
    balance.to_csv(tables_dir / f"causal_propensity_balance{suffix}.csv", index=False)
    logger.info("Covariate balance (max |SMD|): before=%.3f, after=%.3f",
                balance["smd_before_weighting"].abs().max(), balance["smd_after_weighting"].abs().max())
    logger.info("Covariate balance table:\n%s", balance.to_string(index=False))

    # ---- 3. IPTW-weighted Cox model with treatment x cluster interaction -------------------
    cox_df = confounders.copy()
    cox_df["treatment"] = treatment.values
    cox_df["cluster"] = cluster.values
    cox_df["duration"], cox_df["event"], cox_df["weight"] = durations, events, sw_trimmed

    confounder_terms = " + ".join(confounders.columns)
    formula_full = f"treatment * C(cluster) + {confounder_terms}"
    formula_reduced = f"treatment + C(cluster) + {confounder_terms}"

    cph_full = CoxPHFitter(penalizer=0.1)
    cph_full.fit(cox_df, duration_col="duration", event_col="event", weights_col="weight",
                 formula=formula_full, robust=True)
    cph_reduced = CoxPHFitter(penalizer=0.1)
    cph_reduced.fit(cox_df, duration_col="duration", event_col="event", weights_col="weight",
                     formula=formula_reduced, robust=True)

    lr_stat = 2 * (cph_full.log_likelihood_ - cph_reduced.log_likelihood_)
    df_diff = cph_full.params_.shape[0] - cph_reduced.params_.shape[0]
    lrt_p = stats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
    pd.DataFrame([{"lr_statistic": lr_stat, "df": df_diff, "p_value": lrt_p}]).to_csv(
        tables_dir / f"causal_effect_modification_lrt{suffix}.csv", index=False)
    logger.info("Effect-modification LRT: stat=%.3f, df=%d, p=%.4g", lr_stat, df_diff, lrt_p)

    beta, cov = cph_full.params_, cph_full.variance_matrix_
    clusters_sorted = sorted(cluster.unique())
    ref_cluster = clusters_sorted[0]
    rows = []
    for c in clusters_sorted:
        contrast = pd.Series(0.0, index=beta.index)
        contrast["treatment"] = 1.0
        inter_name = f"treatment:C(cluster)[T.{c}]"
        if inter_name in contrast.index:
            contrast[inter_name] = 1.0
        log_hr = float(contrast @ beta)
        se = float(np.sqrt(contrast @ cov @ contrast))
        hr, lo_ci, hi_ci = np.exp(log_hr), np.exp(log_hr - 1.96 * se), np.exp(log_hr + 1.96 * se)
        z = log_hr / se if se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else np.nan
        n_treat = int(((cluster == c) & (treatment == 1)).sum())
        n_ctrl = int(((cluster == c) & (treatment == 0)).sum())
        rows.append({"cluster": c, "is_reference_cluster": c == ref_cluster, "hazard_ratio": hr,
                     "ci_lower": lo_ci, "ci_upper": hi_ci, "p_value": p, "e_value": evalue(hr),
                     "n_treated": n_treat, "n_control": n_ctrl})
    forest_cox = pd.DataFrame(rows)
    forest_cox.to_csv(tables_dir / f"causal_cox_forest_plot{suffix}.csv", index=False)
    logger.info("Cluster-specific chemotherapy HRs (IPTW-weighted Cox):\n%s", forest_cox.to_string(index=False))

    # ---- 4. causal forest on RMST pseudo-outcome -------------------------------------------
    horizon = ccfg["causal_forest"]["rmst_horizon_months"]
    logger.info("Computing jackknife RMST pseudo-observations at horizon=%d months (n=%d) ...",
                horizon, len(cohort))
    Y = jackknife_rmst_pseudo_obs(durations, events, horizon)

    cf = CausalForestDML(
        n_estimators=ccfg["causal_forest"]["n_estimators"],
        min_samples_leaf=ccfg["causal_forest"]["min_samples_leaf"],
        random_state=ccfg["causal_forest"]["random_seed"],
        discrete_treatment=True,
    )
    cf.fit(Y, treatment.values, X=cluster_dummies.values, W=confounders.values)

    dummy_cols = cluster_dummies.columns.tolist()
    X_by_cluster = np.zeros((len(clusters_sorted), len(dummy_cols)))
    for i, c in enumerate(clusters_sorted):
        col = f"cluster_{c}"
        if col in dummy_cols:
            X_by_cluster[i, dummy_cols.index(col)] = 1.0

    cate = cf.effect(X_by_cluster)
    lb, ub = cf.effect_interval(X_by_cluster, alpha=cfg["causal"]["alpha"])
    cate_df = pd.DataFrame([{"cluster": c, "cate_rmst_months": cate[i], "ci_lower": lb[i], "ci_upper": ub[i]}
                            for i, c in enumerate(clusters_sorted)])
    cate_df.to_csv(tables_dir / f"causal_forest_cate_by_cluster{suffix}.csv", index=False)
    logger.info("Causal forest CATE (RMST months gained from chemo) by cluster:\n%s", cate_df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("07_causal_treatment_effects", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])

    ccfg = cfg["causal"]
    clinical = pd.read_parquet(proc_dir / "clinical_with_clusters.parquet")
    covariates = pd.read_parquet(proc_dir / "clinical_covariates.parquet")
    insample_labels = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    train_ids = insample_labels.index

    # ---- out-of-fold cluster labels for the whole training set --------------------------------
    Z_train = pd.read_parquet(proc_dir / "mofa_factors_train.parquet").loc[train_ids]
    dur_all, ev_all = clinical.loc[train_ids, "OS_MONTHS_NUM"].values, clinical.loc[train_ids, "OS_EVENT"].values
    ref_centroids = joblib.load(resolve(cfg["paths"]["checkpoints_dir"]) / "kmeans_model.joblib")["model"].cluster_centers_
    n_folds = ccfg.get("oof_n_folds", 5)
    logger.info("Deriving out-of-fold cluster labels on the training set (%d folds) ...", n_folds)
    oof_labels, fold_summary = out_of_fold_labels(Z_train, dur_all, ev_all, cfg["clustering"], n_folds,
                                                  cfg["project"]["random_seed"], ref_centroids, logger)
    ari = adjusted_rand_score(insample_labels.loc[train_ids], oof_labels.loc[train_ids])
    agree = float((insample_labels.loc[train_ids].values == oof_labels.loc[train_ids].values).mean())
    fold_summary["ari_oof_vs_insample_all_train"] = ari
    fold_summary["label_agreement_oof_vs_insample_all_train"] = agree
    fold_summary.to_csv(tables_dir / "causal_oof_fold_summary.csv", index=False)
    oof_labels.rename("cluster").to_frame().to_parquet(proc_dir / "cluster_labels_train_oof.parquet")
    logger.info("Out-of-fold vs in-sample training labels: ARI=%.3f, exact agreement=%.1f%%; OOF sizes=%s",
                ari, 100 * agree, oof_labels.value_counts().sort_index().to_dict())

    # ---- causal cohort ----------------------------------------------------------------------------
    cohort = clinical.loc[train_ids].copy()
    filt_col, filt_val = next(iter(ccfg["cohort_filter"].items()))
    cohort = cohort[cohort[filt_col] == filt_val]
    cohort = cohort.dropna(subset=[ccfg["treatment_col"], "OS_MONTHS_NUM", "OS_EVENT"])
    cohort = cohort[cohort[ccfg["treatment_col"]] != ""]
    logger.info("Causal cohort (%s=%s, train split): n=%d", filt_col, filt_val, len(cohort))

    # primary: out-of-fold labels; comparison: in-sample stage 05 labels
    run_causal_analysis(cohort, covariates, oof_labels, cfg, tables_dir, logger, suffix="")
    run_causal_analysis(cohort, covariates, insample_labels, cfg, tables_dir, logger, suffix="_insample_labels")

    logger.info("Stage 07 complete.")


if __name__ == "__main__":
    main()
