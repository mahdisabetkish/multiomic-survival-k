#!/usr/bin/env python3
"""Stage 04: multi-omic factorisation with MOFA.

Fits MOFA (mofapy2, CPU-only) on the three TRAIN-only views produced by
stage 03 (mrna_pathway, cna_pathway, mutation), then projects the held-out
test patients into the same latent space.

Leakage control: every view is standardised (zero mean, unit variance per
feature) using statistics fit on the TRAIN split only; MOFA itself is also
trained on TRAIN data only. Test-set factor scores are obtained via a
closed-form ridge/posterior-style projection that reuses MOFA's own
converged loadings (W) and per-feature noise precision (Tau) -- mofapy2 has
no built-in "transform new data" API, so this reproduces MOFA's own
coordinate-ascent update rule for Z with W and Tau held fixed at their
trained values, which is the principled way to place new samples in an
existing factor space without refitting on them.

All three views are modelled with a Gaussian likelihood: the mutation view
mixes binary gene-level indicators with a continuous TMB score, which rules
out a pure Bernoulli likelihood for that view, so Gaussian + per-feature
standardisation is used uniformly across views (a common practical
approximation in applied MOFA analyses).

Outputs:
    data/processed/metabric/mofa_factors_train.parquet
    data/processed/metabric/mofa_factors_test.parquet
    outputs/checkpoints/mofa_model.joblib   (W, Tau, scalers, feature/view
                                             names -- reused by stage 08 to
                                             project the TCGA cohort)
    outputs/tables/mofa_variance_explained.csv
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point

from common import load_config, get_logger, resolve

VIEW_FILES = {
    "mrna_pathway": "view_mrna_pathway.parquet",
    "cna_pathway": "view_cna_pathway.parquet",
    "mutation": "view_mutation.parquet",
}


def project_test_factors(views_test_std: dict[str, np.ndarray], W: dict[str, np.ndarray],
                          tau: dict[str, np.ndarray], n_factors: int) -> np.ndarray:
    """Closed-form projection of new samples into a trained MOFA factor space.

    Reuses MOFA's own conditional-posterior update for Z given (W, Tau) held
    fixed at their converged training values, with a unit-variance prior on
    each factor (identity regulariser), i.e. Bayesian-linear-regression-style
    ridge projection.
    """
    K = n_factors
    precision = np.eye(K)
    rhs_terms = []
    n_samples = next(iter(views_test_std.values())).shape[0]
    rhs = np.zeros((n_samples, K))
    for view in W:
        Wv, tauv = W[view], tau[view]          # Wv: (D,K), tauv: (D,)
        Xv = views_test_std[view]              # (N,D)
        precision += Wv.T @ (Wv * tauv[:, None])
        rhs += Xv @ (Wv * tauv[:, None])
    sigma_z = np.linalg.inv(precision)
    Z_test = rhs @ sigma_z.T
    return Z_test


def variance_explained(views_std: dict[str, np.ndarray], Z: np.ndarray, W: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for view, X in views_std.items():
        ss_tot = np.sum(X ** 2)
        Xhat = Z @ W[view].T
        ss_res = np.sum((X - Xhat) ** 2)
        r2_total = 1 - ss_res / ss_tot
        per_factor = []
        for k in range(Z.shape[1]):
            Xhat_k = np.outer(Z[:, k], W[view][:, k])
            ss_res_k = np.sum((X - Xhat_k) ** 2)
            per_factor.append(1 - ss_res_k / ss_tot)
        rows.append({"view": view, "R2_total": r2_total, **{f"Factor{k+1}_R2": v for k, v in enumerate(per_factor)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("04_mofa_factorisation", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    ckpt_dir = resolve(cfg["paths"]["checkpoints_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    mcfg = cfg["mofa"]
    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    train_ids = clinical.index[clinical["split"] == "train"].tolist()
    test_ids = clinical.index[clinical["split"] == "test"].tolist()

    views_raw = {v: pd.read_parquet(proc_dir / f).reindex(clinical.index) for v, f in VIEW_FILES.items()}

    # ---- standardise each view, fit on train only -----------------------------------------
    scalers, views_train_std, views_test_std, feature_names = {}, {}, {}, {}
    for view, df in views_raw.items():
        scaler = StandardScaler()
        train_std = scaler.fit_transform(df.loc[train_ids].values)
        test_std = scaler.transform(df.loc[test_ids].values)
        scalers[view] = scaler
        views_train_std[view] = train_std
        views_test_std[view] = test_std
        feature_names[view] = df.columns.tolist()
        logger.info("View '%s': %d features, train=%d, test=%d", view, df.shape[1], len(train_ids), len(test_ids))

    # ---- fit MOFA on train only --------------------------------------------------------------
    view_order = mcfg["views"]
    data = [[views_train_std[v]] for v in view_order]

    ep = entry_point()
    ep.set_data_options(scale_views=False, center_groups=False)
    ep.set_data_matrix(
        data, likelihoods=["gaussian"] * len(view_order),
        views_names=view_order, groups_names=["metabric_train"],
        samples_names=[train_ids],
        features_names=[feature_names[v] for v in view_order],
    )
    ep.set_model_options(factors=mcfg["n_factors_default"])
    ep.set_train_options(
        iter=mcfg["max_iterations"], convergence_mode=mcfg["convergence_mode"],
        dropR2=mcfg["drop_factor_threshold"], seed=mcfg["seed"], quiet=False, verbose=False,
    )
    logger.info("Building and training MOFA model (max_factors=%d, dropR2=%.3f) ...",
                mcfg["n_factors_default"], mcfg["drop_factor_threshold"])
    ep.build()
    ep.run()

    Z_train = ep.model.nodes["Z"].getExpectation()
    n_factors = Z_train.shape[1]
    logger.info("MOFA converged with %d active factors (of %d initial)", n_factors, mcfg["n_factors_default"])
    if not (mcfg["n_factors_min"] <= n_factors <= mcfg["n_factors_max"]):
        logger.warning("Retained factor count %d is outside the target range [%d, %d]",
                        n_factors, mcfg["n_factors_min"], mcfg["n_factors_max"])

    W, tau = {}, {}
    for m, view in enumerate(view_order):
        W[view] = ep.model.nodes["W"].nodes[m].getExpectation()
        tau[view] = ep.model.nodes["Tau"].nodes[m].getExpectation().mean(axis=0)

    factor_cols = [f"Factor{k+1}" for k in range(n_factors)]
    Z_train_df = pd.DataFrame(Z_train, index=train_ids, columns=factor_cols)

    Z_test = project_test_factors({v: views_test_std[v] for v in view_order}, W, tau, n_factors)
    Z_test_df = pd.DataFrame(Z_test, index=test_ids, columns=factor_cols)

    Z_train_df.to_parquet(proc_dir / "mofa_factors_train.parquet")
    Z_test_df.to_parquet(proc_dir / "mofa_factors_test.parquet")

    var_exp = variance_explained({v: views_train_std[v] for v in view_order}, Z_train, W)
    var_exp.to_csv(tables_dir / "mofa_variance_explained.csv", index=False)
    logger.info("Variance explained per view:\n%s", var_exp[["view", "R2_total"]].to_string(index=False))

    joblib.dump({
        "view_order": view_order, "n_factors": n_factors, "W": W, "tau": tau,
        "scalers": scalers, "feature_names": feature_names,
        "git_commit": None,
    }, ckpt_dir / "mofa_model.joblib")

    logger.info("Stage 04 complete. Train factors=%s, test factors=%s", Z_train_df.shape, Z_test_df.shape)


if __name__ == "__main__":
    main()
