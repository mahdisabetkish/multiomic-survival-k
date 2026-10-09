#!/usr/bin/env python3
"""Stage 16: can the protocol detect a survival-relevant grouping when one exists?

A negative result is only informative if the procedure that produced it could have
produced a positive one. The evaluation in this paper finds that choosing the number
of clusters by survival separation adds almost nothing over choosing it by an
unsupervised criterion. That could mean the step does not help, or it could mean the
evaluation is not sensitive enough to notice if it did.

This script settles the question on simulated data where the answer is known.

The design follows from what the step can possibly do. Survival is used only to choose
k, so it can only change the partition when the survival-relevant structure has a
different granularity from the structure that dominates the feature covariance. The
simulation therefore contains two independent groupings of the same observations:

    nuisance   two well-separated groups, loading on the first latent factor,
               carrying no information about survival;
    signal     four groups, loading on later factors with a smaller separation,
               which determine the hazard.

An unsupervised criterion should find the two nuisance groups. A survival-guided
criterion should find the four signal groups, if it works. The separation of the
signal groups in feature space is held fixed while the strength of their association
with survival is varied, so that what changes between settings is only how much
survival information is available, not how findable the groups are.

Everything downstream is the pipeline used in the paper: standardise on the training
part, fit MOFA, choose k either by the inner survival search or by silhouette, fit
k-means, assign the held-out part to the training centroids, and score a Cox model
with cluster indicators on the held-out part.

Output: outputs/tables/positive_control.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point

from common import load_config, get_logger, resolve

VIEW_DIMS = {"view_a": 150, "view_b": 150, "view_c": 40}
K_LATENT = 8
K_NUISANCE = 2
K_SIGNAL = 4


def simulate(n, signal, nuisance_sep, signal_sep, cens_scale, rng):
    """Two independent groupings; only the second one drives the hazard."""
    g_nuis = rng.randint(0, K_NUISANCE, n)
    g_sig = rng.randint(0, K_SIGNAL, n)

    Z = rng.normal(size=(n, K_LATENT))
    Z[:, 0] += nuisance_sep * g_nuis                      # dominant, survival-irrelevant
    for j in range(K_SIGNAL - 1):                          # fainter, survival-relevant
        Z[:, 1 + j] += signal_sep * (g_sig == j + 1)

    views = {}
    for name, d in VIEW_DIMS.items():
        W = rng.normal(scale=1.0, size=(K_LATENT, d))
        views[name] = Z @ W + rng.normal(scale=1.0, size=(n, d))

    log_hr = signal * (g_sig - (K_SIGNAL - 1) / 2.0)
    t = rng.exponential(1.0 / np.exp(log_hr))
    c = rng.exponential(cens_scale, n)
    return views, np.minimum(t, c), (t <= c).astype(int), g_sig, g_nuis


def fit_mofa_project(views_tr, views_te, mcfg, seed_offset):
    order = list(views_tr.keys())
    scalers = {v: StandardScaler().fit(views_tr[v]) for v in order}
    std_tr = {v: scalers[v].transform(views_tr[v]) for v in order}
    std_te = {v: scalers[v].transform(views_te[v]) for v in order}
    n = std_tr[order[0]].shape[0]

    ep = entry_point()
    ep.set_data_options(scale_views=False, center_groups=False)
    ep.set_data_matrix([[std_tr[v]] for v in order], likelihoods=["gaussian"] * len(order),
                       views_names=order, groups_names=["g"],
                       samples_names=[[f"s{i}" for i in range(n)]],
                       features_names=[[f"{v}_{i}" for i in range(std_tr[v].shape[1])] for v in order])
    ep.set_model_options(factors=mcfg["n_factors_default"])
    ep.set_train_options(iter=mcfg["max_iterations"], convergence_mode=mcfg["convergence_mode"],
                         dropR2=mcfg["drop_factor_threshold"], seed=mcfg["seed"] + seed_offset,
                         quiet=True, verbose=False)
    ep.build()
    ep.run()

    Z_tr = ep.model.nodes["Z"].getExpectation()
    K = Z_tr.shape[1]
    W = {v: ep.model.nodes["W"].nodes[m].getExpectation() for m, v in enumerate(order)}
    tau = {v: ep.model.nodes["Tau"].nodes[m].getExpectation().mean(axis=0) for m, v in enumerate(order)}
    prec = np.eye(K)
    rhs = np.zeros((std_te[order[0]].shape[0], K))
    for v in order:
        prec += W[v].T @ (W[v] * tau[v][:, None])
        rhs += std_te[v] @ (W[v] * tau[v][:, None])
    return Z_tr, rhs @ np.linalg.inv(prec).T


def choose_k_survival(Z, dur, ev, ccfg):
    best, scores = None, {}
    for k in range(ccfg["k_min"], ccfg["k_max"] + 1):
        rep = []
        for r in range(ccfg["n_inner_repeats"]):
            seed = ccfg["random_seed"] + 1000 * k + r
            tr, va = train_test_split(np.arange(len(Z)), test_size=ccfg["inner_val_fraction"],
                                      stratify=ev, random_state=seed)
            km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=seed).fit(Z[tr])
            lab = km.predict(Z[va])
            if len(np.unique(lab)) < 2:
                rep.append(-np.inf)
                continue
            p = multivariate_logrank_test(dur[va], lab, ev[va]).p_value
            rep.append(-np.log10(max(p, 1e-300)))
        scores[k] = float(np.mean(rep))
    return max(scores, key=scores.get), scores


def choose_k_silhouette(Z, ccfg):
    sc = {k: silhouette_score(Z, KMeans(n_clusters=k, n_init=20,
                                        random_state=ccfg["random_seed"]).fit_predict(Z))
          for k in range(ccfg["k_min"], ccfg["k_max"] + 1)}
    return max(sc, key=sc.get), sc


def evaluate(Z_tr, Z_te, k, dur_tr, ev_tr, dur_te, ev_te, g_nuis_tr, g_sig_te, ccfg):
    km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=ccfg["random_seed"]).fit(Z_tr)
    lab_tr, lab_te = km.labels_, km.predict(Z_te)
    d_tr = pd.get_dummies(lab_tr, prefix="c", drop_first=True).astype(float)
    d_te = pd.get_dummies(lab_te, prefix="c", drop_first=True).reindex(columns=d_tr.columns,
                                                                      fill_value=0.0).astype(float)
    df = d_tr.copy(); df["duration"], df["event"] = dur_tr, ev_tr
    try:
        cph = CoxPHFitter(penalizer=0.1).fit(df, duration_col="duration", event_col="event")
        risk = cph.predict_partial_hazard(d_te).values.ravel()
        c = concordance_index(dur_te, -risk, ev_te)
    except Exception:
        c = np.nan
    return {"c_index": c,
            "ari_vs_signal_groups": adjusted_rand_score(g_sig_te, lab_te),
            "ari_vs_nuisance_groups": adjusted_rand_score(g_nuis_tr, lab_tr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("16_positive_control", resolve(cfg["paths"]["logs_dir"]))
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ccfg, mcfg = cfg["clustering"], cfg["mofa"]

    n_tr, n_te = 1302, 558
    signals = [0.0, 0.4, 0.8, 1.6]
    rows = []

    for signal in signals:
        for rep in range(args.reps):
            rng = np.random.RandomState(5000 + int(signal * 100) + rep)
            views, t, d, g_sig, g_nuis = simulate(n_tr + n_te, signal, nuisance_sep=9.0,
                                                  signal_sep=2.5, cens_scale=3.0, rng=rng)
            sl, st = slice(0, n_tr), slice(n_tr, n_tr + n_te)
            v_tr = {k: v[sl] for k, v in views.items()}
            v_te = {k: v[st] for k, v in views.items()}
            Z_tr, Z_te = fit_mofa_project(v_tr, v_te, mcfg, seed_offset=200 + rep)

            k_surv, _ = choose_k_survival(Z_tr, t[sl], d[sl], ccfg)
            k_sil, _ = choose_k_silhouette(Z_tr, ccfg)
            r_surv = evaluate(Z_tr, Z_te, k_surv, t[sl], d[sl], t[st], d[st],
                              g_nuis[sl], g_sig[st], ccfg)
            r_sil = evaluate(Z_tr, Z_te, k_sil, t[sl], d[sl], t[st], d[st],
                             g_nuis[sl], g_sig[st], ccfg)

            rows.append({"signal": signal, "rep": rep + 1, "censoring_pct": 100 * (1 - d.mean()),
                         "k_survival": k_surv, "k_silhouette": k_sil,
                         "c_index_survival": r_surv["c_index"], "c_index_silhouette": r_sil["c_index"],
                         "ari_signal_survival": r_surv["ari_vs_signal_groups"],
                         "ari_signal_silhouette": r_sil["ari_vs_signal_groups"],
                         "ari_nuisance_survival": r_surv["ari_vs_nuisance_groups"],
                         "ari_nuisance_silhouette": r_sil["ari_vs_nuisance_groups"]})
            logger.info("signal=%.1f rep %d: k_surv=%d k_sil=%d | C-index %.3f vs %.3f | "
                        "ARI to signal groups %.3f vs %.3f",
                        signal, rep + 1, k_surv, k_sil, r_surv["c_index"], r_sil["c_index"],
                        r_surv["ari_vs_signal_groups"], r_sil["ari_vs_signal_groups"])

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "positive_control.csv", index=False)

    logger.info("")
    logger.info("%-8s %-16s %-16s %-16s %-16s", "signal", "k survival", "k silhouette",
                "C-index surv", "C-index silh")
    for sg, grp in df.groupby("signal"):
        logger.info("%-8.1f %-16s %-16s %-16.3f %-16.3f", sg,
                    str(sorted(grp.k_survival.tolist())), str(sorted(grp.k_silhouette.tolist())),
                    grp.c_index_survival.mean(), grp.c_index_silhouette.mean())
    logger.info("Stage 16 complete.")


if __name__ == "__main__":
    main()
