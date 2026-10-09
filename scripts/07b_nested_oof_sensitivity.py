#!/usr/bin/env python3
"""Sensitivity check: fully nested out-of-fold cluster labels.

Stage 07 obtains out-of-fold labels by choosing k and fitting the k-means
centroids inside each fold, so that no patient's label depends on that
patient's own survival time. The latent representation, however, is shared:
the MOFA factors and the per-view standardisation were fitted once on the
whole training set, in stage 04.

MOFA and the standardisation never see survival, so sharing them cannot
reintroduce the circularity that the fold-wise procedure was built to remove.
It is nevertheless a weaker form of nesting than refitting everything inside
each fold. This script does the stricter version: for every fold, the view
scalers and MOFA are refitted on the other folds only, the held-out fold is
projected into that fold's latent space, and k and the centroids are chosen
there. Fold clusters are matched to the stage 05 reference labels by overlap
on the fitting patients, since each fold now has its own latent space and
centroids are not directly comparable across folds.

The causal analysis is then repeated with these labels and compared with the
stage 07 output.

Outputs (outputs/tables/):
    causal_nested_oof_agreement.csv
    causal_cox_forest_plot_nested.csv
    causal_effect_modification_lrt_nested.csv
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point

from common import load_config, get_logger, resolve

VIEW_FILES = {
    "mrna_pathway": "view_mrna_pathway.parquet",
    "cna_pathway": "view_cna_pathway.parquet",
    "mutation": "view_mutation.parquet",
}


def _load_stage07():
    path = Path(__file__).parent / "07_causal_treatment_effects.py"
    spec = importlib.util.spec_from_file_location("stage07", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_mofa(views_std: dict, view_order, mcfg, seed_offset=0):
    n = views_std[view_order[0]].shape[0]
    feat_names = [[f"{v}_{i}" for i in range(views_std[v].shape[1])] for v in view_order]
    data = [[views_std[v]] for v in view_order]
    ep = entry_point()
    ep.set_data_options(scale_views=False, center_groups=False)
    ep.set_data_matrix(data, likelihoods=["gaussian"] * len(view_order), views_names=view_order,
                       groups_names=["g"], samples_names=[[f"s{i}" for i in range(n)]],
                       features_names=feat_names)
    ep.set_model_options(factors=mcfg["n_factors_default"])
    ep.set_train_options(iter=mcfg["max_iterations"], convergence_mode=mcfg["convergence_mode"],
                         dropR2=mcfg["drop_factor_threshold"], seed=mcfg["seed"] + seed_offset,
                         quiet=True, verbose=False)
    ep.build()
    ep.run()
    Z = ep.model.nodes["Z"].getExpectation()
    W = {v: ep.model.nodes["W"].nodes[m].getExpectation() for m, v in enumerate(view_order)}
    tau = {v: ep.model.nodes["Tau"].nodes[m].getExpectation().mean(axis=0) for m, v in enumerate(view_order)}
    return Z, W, tau


def project_mofa(views_std, W, tau, view_order, n_factors):
    precision = np.eye(n_factors)
    rhs = np.zeros((next(iter(views_std.values())).shape[0], n_factors))
    for v in view_order:
        precision += W[v].T @ (W[v] * tau[v][:, None])
        rhs += views_std[v] @ (W[v] * tau[v][:, None])
    return rhs @ np.linalg.inv(precision).T


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


def match_by_overlap(fold_labels_fit, reference_labels_fit, k_fold, k_ref):
    """Map fold cluster ids to reference cluster ids by overlap on the fitting patients.

    Each fold has its own latent space, so centroids cannot be compared directly;
    the overlap table is used instead. This is a relabelling and uses no survival
    information from the held-out patients.
    """
    tab = np.zeros((k_fold, k_ref))
    for a in range(k_fold):
        for b in range(k_ref):
            tab[a, b] = np.sum((fold_labels_fit == a) & (reference_labels_fit == b))
    if k_fold == k_ref:
        rows, cols = linear_sum_assignment(-tab)
        return dict(zip(rows, cols)), False
    return {a: int(tab[a].argmax()) for a in range(k_fold)}, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("07b_nested_oof_sensitivity", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ccfg, mcfg = cfg["clustering"], cfg["mofa"]
    stage07 = _load_stage07()

    clinical = pd.read_parquet(proc_dir / "clinical_with_clusters.parquet")
    covariates = pd.read_parquet(proc_dir / "clinical_covariates.parquet")
    reference = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    shared_oof = pd.read_parquet(proc_dir / "cluster_labels_train_oof.parquet")["cluster"]
    train_ids = reference.index
    k_ref = reference.nunique()

    views_raw = {v: pd.read_parquet(proc_dir / f).loc[train_ids] for v, f in VIEW_FILES.items()}
    view_order = mcfg["views"]
    dur = clinical.loc[train_ids, "OS_MONTHS_NUM"].values
    ev = clinical.loc[train_ids, "OS_EVENT"].values

    n_folds = cfg["causal"].get("oof_n_folds", 5)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg["project"]["random_seed"])
    labels = pd.Series(-1, index=train_ids, dtype=int)
    rows = []

    for f, (fit_idx, held_idx) in enumerate(skf.split(np.zeros(len(train_ids)), ev), 1):
        logger.info("Fold %d/%d: refitting scalers and MOFA on %d patients, projecting %d",
                    f, n_folds, len(fit_idx), len(held_idx))
        scalers, v_fit, v_held = {}, {}, {}
        for v in view_order:
            sc = StandardScaler().fit(views_raw[v].values[fit_idx])
            scalers[v] = sc
            v_fit[v] = sc.transform(views_raw[v].values[fit_idx])
            v_held[v] = sc.transform(views_raw[v].values[held_idx])

        Z_fit, W, tau = fit_mofa(v_fit, view_order, mcfg, seed_offset=10 + f)
        Z_held = project_mofa(v_held, W, tau, view_order, Z_fit.shape[1])

        k_f, scores = inner_k_search(Z_fit, dur[fit_idx], ev[fit_idx], ccfg)
        km = KMeans(n_clusters=k_f, n_init=ccfg["n_init_per_k"],
                    random_state=ccfg["random_seed"]).fit(Z_fit)
        mapping, merged = match_by_overlap(km.labels_, reference.values[fit_idx], k_f, k_ref)
        labels.iloc[held_idx] = [mapping[c] for c in km.predict(Z_held)]
        rows.append({"fold": f, "n_fit": len(fit_idx), "n_held_out": len(held_idx),
                     "n_factors": Z_fit.shape[1], "k_selected": k_f,
                     "clusters_merged_on_matching": merged})
        logger.info("Fold %d: MOFA kept %d factors, k*=%d%s", f, Z_fit.shape[1], k_f,
                    " (k differs from reference)" if merged else "")

    assert (labels >= 0).all()
    fold_df = pd.DataFrame(rows)
    ari_ref = adjusted_rand_score(reference.loc[train_ids], labels.loc[train_ids])
    ari_shared = adjusted_rand_score(shared_oof.loc[train_ids], labels.loc[train_ids])
    agree_shared = float((shared_oof.loc[train_ids].values == labels.loc[train_ids].values).mean())
    fold_df["ari_vs_reference"] = ari_ref
    fold_df["ari_vs_shared_representation_oof"] = ari_shared
    fold_df["agreement_vs_shared_representation_oof"] = agree_shared
    fold_df.to_csv(tables_dir / "causal_nested_oof_agreement.csv", index=False)
    logger.info("Fully nested labels vs stage 07 out-of-fold labels: ARI=%.3f, agreement=%.1f%%",
                ari_shared, 100 * agree_shared)
    logger.info("Fully nested labels vs reference training labels:   ARI=%.3f", ari_ref)

    cohort = clinical.loc[train_ids].copy()
    fcol, fval = next(iter(cfg["causal"]["cohort_filter"].items()))
    cohort = cohort[cohort[fcol] == fval]
    cohort = cohort.dropna(subset=[cfg["causal"]["treatment_col"], "OS_MONTHS_NUM", "OS_EVENT"])
    cohort = cohort[cohort[cfg["causal"]["treatment_col"]] != ""]
    logger.info("Repeating the causal analysis on %d ER-positive training patients", len(cohort))
    stage07.run_causal_analysis(cohort, covariates, labels, cfg, tables_dir, logger,
                                suffix="_nested")
    logger.info("Stage 07b complete.")


if __name__ == "__main__":
    main()
