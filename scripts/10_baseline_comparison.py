#!/usr/bin/env python3
"""Stage 10: comparison against alternative multi-omic clustering methods.

Evaluates SAMOC (this paper's method) against:
  - standard/vanilla MOFA + k-means: same MOFA factors as SAMOC, but with k
    and the partition chosen by unsupervised silhouette score instead of
    survival-aware inner-CV selection. Isolates the contribution of the
    survival-aware selection step (also reported as an ablation, stage 11).
  - SNF (Similarity Network Fusion, Wang et al. 2014): run for real via the
    `snfpy` package (genuine third-party implementation, not a
    reimplementation).
  - NEMO (Rappoport & Shamir 2019) and CIMLR (Ramazzotti et al. 2018): both
    are R/Bioconductor-only with no maintained Python port. We implement
    Python reimplementations of their documented core algorithms (self-tuned
    per-view affinity averaging + spectral clustering for NEMO; multi-kernel
    per-view affinity averaging + spectral clustering for CIMLR's simplified
    fusion step). These are NOT the original packages and do not reproduce
    their exact optimization procedures (NEMO's is otherwise a simple
    average so this is a faithful reproduction; CIMLR's alternating
    multi-kernel-weight learning and rank-constrained fusion are
    approximated by uniform kernel averaging). This is stated explicitly
    here and in the manuscript -- results for these two rows should be read
    as "a python approximation of [method]", not as the original tool.
  - iClusterPlus (Shen et al. 2009, 2012): also R/Bioconductor-only. We
    reimplement the reduced-form joint Gaussian latent variable model that
    the method is built on (concatenated per-view standardization -> joint
    PCA -> k-means), omitting the lasso sparsity penalty and EM estimation
    of the original.

All methods are fit on the TRAINING split only. SNF/NEMO/CIMLR are natively
transductive (graph-based, no inductive out-of-sample mechanism), so test
patients are assigned via the same rule for every such method: nearest
training-cluster centroid in the standardized, concatenated feature space.
This is a necessary adaptation for our leakage-free train/test protocol and
is applied identically across methods for a fair comparison; it is not how
these methods are used in their original (usually train+test-together)
publications.

Outputs (outputs/tables/):
    baseline_comparison.csv
    baseline_significance_tests.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# snfpy (last released for an older scikit-learn) calls sklearn.utils.check_array
# with the since-renamed `force_all_finite` kwarg (now `ensure_all_finite`).
# Shim it so we can still run the real, upstream snfpy implementation of SNF
# rather than substituting a reimplementation for a method that does have a
# maintained Python package.
import sklearn.utils as _sk_utils
import sklearn.utils.validation as _sk_validation
_orig_check_array = _sk_validation.check_array
def _check_array_compat(*a, **kw):
    if "force_all_finite" in kw:
        kw["ensure_all_finite"] = kw.pop("force_all_finite")
    return _orig_check_array(*a, **kw)
_sk_utils.check_array = _check_array_compat
_sk_validation.check_array = _check_array_compat

import snf
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from common import load_config, get_logger, resolve

VIEW_FILES = {
    "mrna_pathway": "view_mrna_pathway.parquet",
    "cna_pathway": "view_cna_pathway.parquet",
    "mutation": "view_mutation.parquet",
}


def cox_eval(train_labels, test_labels, durations_train, events_train, durations_test, events_test):
    dummies_train = pd.get_dummies(train_labels, prefix="c", drop_first=True).astype(float)
    dummies_test = pd.get_dummies(test_labels, prefix="c", drop_first=True).reindex(
        columns=dummies_train.columns, fill_value=0.0).astype(float)
    df = dummies_train.copy()
    df["duration"], df["event"] = durations_train, events_train
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(df, duration_col="duration", event_col="event")
        risk_test = cph.predict_partial_hazard(dummies_test).values.ravel()
        c_index = concordance_index(durations_test, -risk_test, events_test)
    except Exception:
        risk_test = np.zeros(len(test_labels))
        c_index = np.nan
    n_k = len(np.unique(np.concatenate([train_labels, test_labels])))
    logrank_p = multivariate_logrank_test(durations_test, test_labels, events_test).p_value if n_k > 1 else np.nan
    return c_index, logrank_p, risk_test


def nearest_centroid_assign(X_train, train_labels, X_test):
    centroids = np.array([X_train[train_labels == c].mean(axis=0) for c in sorted(np.unique(train_labels))])
    d = euclidean_distances(X_test, centroids)
    return np.array(sorted(np.unique(train_labels)))[d.argmin(axis=1)]


def self_tuned_affinity(X, k_local=10):
    """SNF/NEMO-style self-tuning Gaussian affinity: bandwidth per point set
    to its mean distance to its k_local nearest neighbours (Zelnik-Manor &
    Perona 2004 local scaling, the construction both SNF and NEMO use)."""
    d = euclidean_distances(X)
    n = d.shape[0]
    sigma = np.array([np.sort(d[i])[1:k_local + 1].mean() for i in range(n)])
    sigma = np.maximum(sigma, 1e-6)
    aff = np.exp(-(d ** 2) / (sigma[:, None] * sigma[None, :]))
    np.fill_diagonal(aff, 0.0)
    return aff


def multi_kernel_affinity(X, sigmas_pct=(10, 25, 50, 75, 90)):
    """CIMLR-style multi-kernel construction: several Gaussian kernels at
    different bandwidths (percentiles of the pairwise distance distribution)
    per view, uniformly averaged (simplified stand-in for CIMLR's learned
    kernel weights)."""
    d = euclidean_distances(X)
    tri = d[np.triu_indices_from(d, k=1)]
    kernels = []
    for p in sigmas_pct:
        sigma = max(np.percentile(tri, p), 1e-6)
        k = np.exp(-(d ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(k, 0.0)
        kernels.append(k)
    return np.mean(kernels, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("10_baseline_comparison", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    seed = cfg["project"]["random_seed"]

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    Z_train = pd.read_parquet(proc_dir / "mofa_factors_train.parquet")
    Z_test = pd.read_parquet(proc_dir / "mofa_factors_test.parquet")
    samoc_train = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    samoc_test = pd.read_parquet(proc_dir / "cluster_labels_test.parquet")["cluster"]
    K_MATCH = samoc_train.nunique()

    train_ids, test_ids = Z_train.index, Z_test.index
    dur_tr, ev_tr = clinical.loc[train_ids, "OS_MONTHS_NUM"].values, clinical.loc[train_ids, "OS_EVENT"].values
    dur_te, ev_te = clinical.loc[test_ids, "OS_MONTHS_NUM"].values, clinical.loc[test_ids, "OS_EVENT"].values

    # concatenated, standardized raw view features (train-fit scaler) -- used by every baseline
    views_raw = {v: pd.read_parquet(proc_dir / f) for v, f in VIEW_FILES.items()}
    scaler = StandardScaler().fit(pd.concat([views_raw[v].loc[train_ids] for v in VIEW_FILES], axis=1).values)
    X_train_concat = scaler.transform(pd.concat([views_raw[v].loc[train_ids] for v in VIEW_FILES], axis=1).values)
    X_test_concat = scaler.transform(pd.concat([views_raw[v].loc[test_ids] for v in VIEW_FILES], axis=1).values)
    per_view_train = {v: StandardScaler().fit_transform(views_raw[v].loc[train_ids].values) for v in VIEW_FILES}
    per_view_test = {v: StandardScaler().fit(views_raw[v].loc[train_ids].values).transform(views_raw[v].loc[test_ids].values)
                      for v in VIEW_FILES}

    results = []

    # ---- SAMOC (this paper's method): reference row -----------------------------------------
    c, p, risk_samoc = cox_eval(samoc_train.values, samoc_test.values, dur_tr, ev_tr, dur_te, ev_te)
    results.append({"method": "SAMOC (this paper)", "k": K_MATCH, "c_index": c, "logrank_p": p,
                     "is_reimplementation": False, "notes": "survival-aware MOFA clustering, real"})
    logger.info("SAMOC: C-index=%.4f, logrank_p=%.4g", c, p)

    # ---- standard/vanilla MOFA + k-means (unsupervised k via silhouette) --------------------
    sil_scores = {}
    for k in range(2, 9):
        labels_k = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(Z_train.values)
        sil_scores[k] = silhouette_score(Z_train.values, labels_k)
    best_k = max(sil_scores, key=sil_scores.get)
    km = KMeans(n_clusters=best_k, n_init=50, random_state=seed).fit(Z_train.values)
    vanilla_train, vanilla_test = km.labels_, km.predict(Z_test.values)
    c, p, risk_vanilla = cox_eval(vanilla_train, vanilla_test, dur_tr, ev_tr, dur_te, ev_te)
    results.append({"method": "Standard MOFA + k-means (unsupervised k)", "k": best_k, "c_index": c, "logrank_p": p,
                     "is_reimplementation": False, "notes": f"k chosen by silhouette (scores={sil_scores})"})
    logger.info("Vanilla MOFA+kmeans: k=%d, C-index=%.4f, logrank_p=%.4g", best_k, c, p)

    # ---- SNF (real, snfpy) -------------------------------------------------------------------
    affinities = snf.make_affinity([per_view_train[v] for v in VIEW_FILES], metric="sqeuclidean", K=20, mu=0.5)
    fused = snf.snf(affinities, K=20)
    snf_train = SpectralClustering(n_clusters=K_MATCH, affinity="precomputed", random_state=seed).fit_predict(fused)
    snf_test = nearest_centroid_assign(X_train_concat, snf_train, X_test_concat)
    c, p, risk_snf = cox_eval(snf_train, snf_test, dur_tr, ev_tr, dur_te, ev_te)
    results.append({"method": "SNF (Wang et al. 2014)", "k": K_MATCH, "c_index": c, "logrank_p": p,
                     "is_reimplementation": False, "notes": "real snfpy package, spectral clustering, k matched to SAMOC"})
    logger.info("SNF: C-index=%.4f, logrank_p=%.4g", c, p)

    # ---- NEMO reimplementation ----------------------------------------------------------------
    nemo_affinities = [self_tuned_affinity(per_view_train[v]) for v in VIEW_FILES]
    nemo_fused = np.mean(nemo_affinities, axis=0)
    nemo_train = SpectralClustering(n_clusters=K_MATCH, affinity="precomputed", random_state=seed).fit_predict(nemo_fused)
    nemo_test = nearest_centroid_assign(X_train_concat, nemo_train, X_test_concat)
    c, p, risk_nemo = cox_eval(nemo_train, nemo_test, dur_tr, ev_tr, dur_te, ev_te)
    results.append({"method": "NEMO-repro (Rappoport & Shamir 2019)", "k": K_MATCH, "c_index": c, "logrank_p": p,
                     "is_reimplementation": True, "notes": "python reimplementation: self-tuned per-view affinity, unweighted average, spectral clustering"})
    logger.info("NEMO-repro: C-index=%.4f, logrank_p=%.4g", c, p)

    # ---- CIMLR reimplementation ----------------------------------------------------------------
    cimlr_affinities = [multi_kernel_affinity(per_view_train[v]) for v in VIEW_FILES]
    cimlr_fused = np.mean(cimlr_affinities, axis=0)
    cimlr_train = SpectralClustering(n_clusters=K_MATCH, affinity="precomputed", random_state=seed).fit_predict(cimlr_fused)
    cimlr_test = nearest_centroid_assign(X_train_concat, cimlr_train, X_test_concat)
    c, p, risk_cimlr = cox_eval(cimlr_train, cimlr_test, dur_tr, ev_tr, dur_te, ev_te)
    results.append({"method": "CIMLR-repro (Ramazzotti et al. 2018)", "k": K_MATCH, "c_index": c, "logrank_p": p,
                     "is_reimplementation": True, "notes": "python reimplementation: multi-kernel per-view affinity averaging, spectral clustering"})
    logger.info("CIMLR-repro: C-index=%.4f, logrank_p=%.4g", c, p)

    # ---- iClusterPlus reimplementation ---------------------------------------------------------
    pca = PCA(n_components=Z_train.shape[1], random_state=seed).fit(X_train_concat)
    icp_Z_train, icp_Z_test = pca.transform(X_train_concat), pca.transform(X_test_concat)
    icp_km = KMeans(n_clusters=K_MATCH, n_init=50, random_state=seed).fit(icp_Z_train)
    icp_train, icp_test = icp_km.labels_, icp_km.predict(icp_Z_test)
    c, p, risk_icp = cox_eval(icp_train, icp_test, dur_tr, ev_tr, dur_te, ev_te)
    results.append({"method": "iClusterPlus-repro (Shen et al. 2009/2012)", "k": K_MATCH, "c_index": c, "logrank_p": p,
                     "is_reimplementation": True, "notes": "python reimplementation: concatenated joint PCA (no lasso/EM), k-means"})
    logger.info("iClusterPlus-repro: C-index=%.4f, logrank_p=%.4g", c, p)

    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / "baseline_comparison.csv", index=False)

    # ---- bootstrap significance tests: SAMOC vs every baseline + vs clinical -----------------
    rng = np.random.RandomState(seed)
    n_test = len(test_ids)
    B = 2000
    risk_by_method = {"Standard MOFA + k-means": risk_vanilla, "SNF": risk_snf, "NEMO-repro": risk_nemo,
                       "CIMLR-repro": risk_cimlr, "iClusterPlus-repro": risk_icp}

    sig_rows = []
    for name, risk_other in risk_by_method.items():
        deltas = []
        for _ in range(B):
            idx = rng.randint(0, n_test, n_test)
            c_samoc = concordance_index(dur_te[idx], -risk_samoc[idx], ev_te[idx])
            c_other = concordance_index(dur_te[idx], -risk_other[idx], ev_te[idx])
            deltas.append(c_samoc - c_other)
        deltas = np.array(deltas)
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        p_val = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
        p_val = min(p_val, 1.0)
        sig_rows.append({"comparison": f"SAMOC vs {name}", "delta_c_index": deltas.mean(),
                          "ci_lower": ci_lo, "ci_upper": ci_hi, "p_value": p_val, "n_bootstrap": B})
        logger.info("Bootstrap SAMOC vs %s: delta C-index=%.4f [%.4f, %.4f], p=%.4g",
                    name, deltas.mean(), ci_lo, ci_hi, p_val)

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(tables_dir / "baseline_significance_tests.csv", index=False)

    logger.info("Stage 10 complete.")


if __name__ == "__main__":
    main()
