#!/usr/bin/env python3
"""Stage 11: ablation study -- contribution of each SAMOC component.

Each variant below runs the SAME outer/inner survival-aware k-means search
procedure as the full method (stage 05), with exactly one component removed
or replaced, then is evaluated identically on the held-out test set:

  - Full SAMOC (reference): pathway-aggregated features -> MOFA -> survival-
    aware k selection.
  - No pathway aggregation: raw high-variance mRNA/CNA genes (top 2000 by
    train variance each) instead of ssGSEA/arm-burden pathway scores,
    otherwise identical (MOFA -> survival-aware k selection). Isolates the
    contribution of biologically-motivated feature aggregation.
  - No MOFA: the survival-aware k-means search runs directly on the
    concatenated, standardized pathway features (no latent factorization).
    Isolates the contribution of MOFA's joint dimensionality reduction.
  - No survival-aware clustering: MOFA factors + unsupervised silhouette-
    selected k-means (identical to stage 10's "standard MOFA + k-means"
    baseline, recomputed here for a self-contained ablation table). Isolates
    the contribution of the survival-aware selection step itself.

The causal treatment-effect-heterogeneity module (stage 07) is not part of
this table: it is not a clustering-performance component (it consumes fixed
cluster labels rather than contributing to their discovery), so its
"ablation" is a capability comparison (with vs. without a formal, testable
treatment-heterogeneity analysis), discussed narratively in the manuscript
rather than forced into a C-index row.

Output: outputs/tables/ablation_study.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point

from common import load_config, get_logger, resolve

VIEW_FILES = {
    "mrna_pathway": "view_mrna_pathway.parquet",
    "cna_pathway": "view_cna_pathway.parquet",
    "mutation": "view_mutation.parquet",
}


def cox_eval(train_labels, test_labels, dur_tr, ev_tr, dur_te, ev_te):
    dummies_train = pd.get_dummies(train_labels, prefix="c", drop_first=True).astype(float)
    dummies_test = pd.get_dummies(test_labels, prefix="c", drop_first=True).reindex(
        columns=dummies_train.columns, fill_value=0.0).astype(float)
    df = dummies_train.copy()
    df["duration"], df["event"] = dur_tr, ev_tr
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col="duration", event_col="event")
    risk_test = cph.predict_partial_hazard(dummies_test).values.ravel()
    c_index = concordance_index(dur_te, -risk_test, ev_te)
    logrank_p = multivariate_logrank_test(dur_te, test_labels, ev_te).p_value
    return c_index, logrank_p


def survival_aware_k_search(Z_train, dur_tr, ev_tr, ccfg, logger, tag):
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
    logger.info("[%s] k search: %s -> k*=%d", tag, results, best_k)
    final_km = KMeans(n_clusters=best_k, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(Z_train)
    return final_km, best_k


def fit_mofa(views_train_std: dict, view_order, mcfg, seed_offset=0):
    feat_names = [[f"{v}_{i}" for i in range(views_train_std[v].shape[1])] for v in view_order]
    data = [[views_train_std[v]] for v in view_order]
    ep = entry_point()
    ep.set_data_options(scale_views=False, center_groups=False)
    ep.set_data_matrix(data, likelihoods=["gaussian"] * len(view_order), views_names=view_order,
                        groups_names=["g"], samples_names=[[f"s{i}" for i in range(views_train_std[view_order[0]].shape[0])]],
                        features_names=feat_names)
    ep.set_model_options(factors=mcfg["n_factors_default"])
    ep.set_train_options(iter=mcfg["max_iterations"], convergence_mode=mcfg["convergence_mode"],
                          dropR2=mcfg["drop_factor_threshold"], seed=mcfg["seed"] + seed_offset, quiet=True, verbose=False)
    ep.build()
    ep.run()
    Z = ep.model.nodes["Z"].getExpectation()
    W = {v: ep.model.nodes["W"].nodes[m].getExpectation() for m, v in enumerate(view_order)}
    tau = {v: ep.model.nodes["Tau"].nodes[m].getExpectation().mean(axis=0) for m, v in enumerate(view_order)}
    return Z, W, tau


def project_mofa(views_test_std, W, tau, view_order, n_factors):
    precision = np.eye(n_factors)
    rhs = np.zeros((next(iter(views_test_std.values())).shape[0], n_factors))
    for v in view_order:
        precision += W[v].T @ (W[v] * tau[v][:, None])
        rhs += views_test_std[v] @ (W[v] * tau[v][:, None])
    return rhs @ np.linalg.inv(precision).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("11_ablation_study", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ccfg, mcfg, seed = cfg["clustering"], cfg["mofa"], cfg["project"]["random_seed"]

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    train_ids = clinical.index[clinical["split"] == "train"]
    test_ids = clinical.index[clinical["split"] == "test"]
    dur_tr, ev_tr = clinical.loc[train_ids, "OS_MONTHS_NUM"].values, clinical.loc[train_ids, "OS_EVENT"].values
    dur_te, ev_te = clinical.loc[test_ids, "OS_MONTHS_NUM"].values, clinical.loc[test_ids, "OS_EVENT"].values

    rows = []

    # ---- Full SAMOC (reference) --------------------------------------------------------------
    samoc_train = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    samoc_test = pd.read_parquet(proc_dir / "cluster_labels_test.parquet")["cluster"]
    c, p = cox_eval(samoc_train.values, samoc_test.values, dur_tr, ev_tr, dur_te, ev_te)
    rows.append({"variant": "Full SAMOC (pathway agg. + MOFA + survival-aware k)", "k": samoc_train.nunique(),
                 "c_index_test": c, "logrank_p_test": p})
    logger.info("Full SAMOC: C-index=%.4f, logrank_p=%.4g", c, p)

    # ---- No survival-aware clustering (MOFA + unsupervised silhouette k-means) --------------
    Z_train = pd.read_parquet(proc_dir / "mofa_factors_train.parquet").values
    Z_test = pd.read_parquet(proc_dir / "mofa_factors_test.parquet").values
    sil = {k: silhouette_score(Z_train, KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(Z_train))
           for k in range(2, 9)}
    best_k = max(sil, key=sil.get)
    km = KMeans(n_clusters=best_k, n_init=50, random_state=seed).fit(Z_train)
    c, p = cox_eval(km.labels_, km.predict(Z_test), dur_tr, ev_tr, dur_te, ev_te)
    rows.append({"variant": "No survival-aware clustering (unsupervised k)", "k": best_k,
                 "c_index_test": c, "logrank_p_test": p})
    logger.info("No-survival-aware: k=%d, C-index=%.4f, logrank_p=%.4g", best_k, c, p)

    # ---- No MOFA (survival-aware k-means directly on standardized pathway features) ---------
    views_raw = {v: pd.read_parquet(proc_dir / f) for v, f in VIEW_FILES.items()}
    scaler_concat = StandardScaler().fit(pd.concat([views_raw[v].loc[train_ids] for v in VIEW_FILES], axis=1).values)
    X_train_concat = scaler_concat.transform(pd.concat([views_raw[v].loc[train_ids] for v in VIEW_FILES], axis=1).values)
    X_test_concat = scaler_concat.transform(pd.concat([views_raw[v].loc[test_ids] for v in VIEW_FILES], axis=1).values)
    km_nomofa, k_nomofa = survival_aware_k_search(X_train_concat, dur_tr, ev_tr, ccfg, logger, "no-MOFA")
    c, p = cox_eval(km_nomofa.labels_, km_nomofa.predict(X_test_concat), dur_tr, ev_tr, dur_te, ev_te)
    rows.append({"variant": "No MOFA (raw concatenated pathway features + survival-aware k)", "k": k_nomofa,
                 "c_index_test": c, "logrank_p_test": p})
    logger.info("No-MOFA: k=%d, C-index=%.4f, logrank_p=%.4g", k_nomofa, c, p)

    # ---- No pathway aggregation (raw high-variance genes -> MOFA -> survival-aware k) -------
    logger.info("Building raw high-variance gene views (no ssGSEA/arm-burden aggregation) ...")
    mrna_raw = pd.read_parquet(proc_dir / "mrna_raw.parquet")
    cna_raw = pd.read_parquet(proc_dir / "cna_raw.parquet")
    top_mrna = mrna_raw.loc[train_ids].var(axis=0).nlargest(2000).index
    top_cna = cna_raw.loc[train_ids].var(axis=0).nlargest(2000).index
    mut_view = pd.read_parquet(proc_dir / VIEW_FILES["mutation"])

    # a small number of gene x patient cells are missing in the raw matrices (~0.02% of mRNA,
    # ~0.7% of CNA); ssGSEA and the pathway-level aggregation both handle this internally, but
    # the raw-gene ablation path needs an explicit fill: gene train-mean for mRNA, 0 (=neutral
    # copy-number call) for CNA
    mrna_sel = mrna_raw[top_mrna].fillna(mrna_raw.loc[train_ids, top_mrna].mean())
    cna_sel = cna_raw[top_cna].fillna(0.0)

    raw_views = {"mrna_raw": mrna_sel, "cna_raw": cna_sel, "mutation": mut_view}
    view_order_raw = list(raw_views.keys())
    scalers_raw = {v: StandardScaler().fit(raw_views[v].loc[train_ids].values) for v in view_order_raw}
    views_train_std = {v: scalers_raw[v].transform(raw_views[v].loc[train_ids].values) for v in view_order_raw}
    views_test_std = {v: scalers_raw[v].transform(raw_views[v].loc[test_ids].values) for v in view_order_raw}

    logger.info("Fitting MOFA on raw high-variance gene views ...")
    Z_raw_train, W_raw, tau_raw = fit_mofa(views_train_std, view_order_raw, mcfg, seed_offset=1)
    Z_raw_test = project_mofa(views_test_std, W_raw, tau_raw, view_order_raw, Z_raw_train.shape[1])
    logger.info("No-pathway-aggregation MOFA retained %d factors", Z_raw_train.shape[1])

    km_nopath, k_nopath = survival_aware_k_search(Z_raw_train, dur_tr, ev_tr, ccfg, logger, "no-pathway-agg")
    c, p = cox_eval(km_nopath.labels_, km_nopath.predict(Z_raw_test), dur_tr, ev_tr, dur_te, ev_te)
    rows.append({"variant": "No pathway aggregation (raw top-2000-variance genes + MOFA + survival-aware k)",
                 "k": k_nopath, "c_index_test": c, "logrank_p_test": p})
    logger.info("No-pathway-aggregation: k=%d, C-index=%.4f, logrank_p=%.4g", k_nopath, c, p)

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(tables_dir / "ablation_study.csv", index=False)
    logger.info("Ablation study:\n%s", ablation_df.to_string(index=False))
    logger.info("Stage 11 complete.")


if __name__ == "__main__":
    main()
