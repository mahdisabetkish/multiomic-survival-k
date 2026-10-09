#!/usr/bin/env python3
"""Stage 18: do the results depend on which gene sets were kept?

KEGG and Reactome were reduced to 150 terms each, drawn at random from the
size-filtered collections of about 320 and 1,800. That keeps the feature count and
the ssGSEA runtime manageable, but it introduces an arbitrary element into the
pipeline that nothing else in the paper examines: a different draw gives a different
feature set, and in principle a different answer.

This script repeats the discovery pipeline under several draws. For each seed it
rebuilds the gene-set collection, recomputes the mRNA pathway scores and the
pathway-level copy-number scores, refits the scalers and MOFA on the training split,
reselects k, refits k-means, assigns the held-out test patients to the training
centroids and scores a Cox model on them. The arm-level copy-number features and the
mutation view do not depend on the draw and are reused.

Outputs: outputs/tables/geneset_seed_sensitivity.csv
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import gseapy as gp
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mofapy2.run.entry_point import entry_point

from common import load_config, get_logger, resolve


def _stage03():
    path = Path(__file__).parent / "03_feature_aggregation.py"
    spec = importlib.util.spec_from_file_location("stage03", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_sets_with_seed(cfg, genesets_dir, seed, logger, s3):
    """stage 03's gene-set construction, with the subsampling seed overridden."""
    ssgsea = cfg["feature_aggregation"]["ssgsea"]
    lo, hi = ssgsea["min_geneset_size"], ssgsea["max_geneset_size"]
    rng = np.random.RandomState(seed)
    combined = {}
    for gs in cfg["feature_aggregation"]["genesets"]:
        name, max_terms = gs["name"], gs.get("max_terms")
        raw = s3.read_gmt(genesets_dir / f"{name}.gmt")
        filtered = {f"{name.upper()}__{t}": g for t, g in raw.items() if lo <= len(g) <= hi}
        if max_terms and len(filtered) > max_terms:
            keys = sorted(filtered.keys())
            keep = rng.choice(keys, size=max_terms, replace=False)
            filtered = {k: filtered[k] for k in keep}
        combined.update(filtered)
    return combined


def ssgsea_scores(mrna, gene_sets, cfg, logger):
    s = cfg["feature_aggregation"]["ssgsea"]
    batch = s.get("sample_batch_size") or mrna.shape[0]
    ids = list(mrna.index)
    chunks = []
    for i in range(0, len(ids), batch):
        expr = mrna.loc[ids[i:i + batch]].T
        expr.index.name = "Gene"
        res = gp.ssgsea(data=expr, gene_sets=gene_sets, outdir=None,
                        sample_norm_method=s["sample_norm_method"],
                        min_size=s["min_geneset_size"], max_size=s["max_geneset_size"],
                        threads=s["threads"], no_plot=True)
        chunks.append(res.res2d.pivot(index="Name", columns="Term", values="NES").astype(float))
        del res, expr
        gc.collect()
    return pd.concat(chunks, axis=0).reindex(mrna.index)


def fit_mofa_project(std_tr, std_te, view_order, mcfg, seed_offset):
    n = std_tr[view_order[0]].shape[0]
    ep = entry_point()
    ep.set_data_options(scale_views=False, center_groups=False)
    ep.set_data_matrix([[std_tr[v]] for v in view_order],
                       likelihoods=["gaussian"] * len(view_order),
                       views_names=view_order, groups_names=["g"],
                       samples_names=[[f"s{i}" for i in range(n)]],
                       features_names=[[f"{v}_{i}" for i in range(std_tr[v].shape[1])]
                                       for v in view_order])
    ep.set_model_options(factors=mcfg["n_factors_default"])
    ep.set_train_options(iter=mcfg["max_iterations"], convergence_mode=mcfg["convergence_mode"],
                         dropR2=mcfg["drop_factor_threshold"], seed=mcfg["seed"] + seed_offset,
                         quiet=True, verbose=False)
    ep.build()
    ep.run()
    Z = ep.model.nodes["Z"].getExpectation()
    K = Z.shape[1]
    W = {v: ep.model.nodes["W"].nodes[m].getExpectation() for m, v in enumerate(view_order)}
    tau = {v: ep.model.nodes["Tau"].nodes[m].getExpectation().mean(axis=0) for m, v in enumerate(view_order)}
    prec = np.eye(K)
    rhs = np.zeros((std_te[view_order[0]].shape[0], K))
    for v in view_order:
        prec += W[v].T @ (W[v] * tau[v][:, None])
        rhs += std_te[v] @ (W[v] * tau[v][:, None])
    return Z, rhs @ np.linalg.inv(prec).T


def select_k(Z, dur, ev, ccfg):
    scores = {}
    for k in range(ccfg["k_min"], ccfg["k_max"] + 1):
        rep = []
        for r in range(ccfg["n_inner_repeats"]):
            seed = ccfg["random_seed"] + 1000 * k + r
            tr, va = train_test_split(np.arange(len(Z)), test_size=ccfg["inner_val_fraction"],
                                      stratify=ev, random_state=seed)
            km = KMeans(n_clusters=k, n_init=ccfg["n_init_per_k"], random_state=seed).fit(Z[tr])
            lab = km.predict(Z[va])
            if len(np.unique(lab)) < 2:
                rep.append(-np.inf); continue
            p = multivariate_logrank_test(dur[va], lab, ev[va]).p_value
            rep.append(-np.log10(max(p, 1e-300)))
        scores[k] = float(np.mean(rep))
    return max(scores, key=scores.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("18_geneset_seed_sensitivity", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    genesets_dir = resolve(cfg["paths"]["genesets_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ccfg, mcfg = cfg["clustering"], cfg["mofa"]
    s3 = _stage03()

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    mrna = pd.read_parquet(proc_dir / "mrna_raw.parquet")
    cna = pd.read_parquet(proc_dir / "cna_raw.parquet")
    mut = pd.read_parquet(proc_dir / "view_mutation.parquet")
    ref = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    train_ids = clinical.index[clinical["split"] == "train"]
    test_ids = clinical.index[clinical["split"] == "test"]

    # the low-variance gene filter is fit on train only, as in stage 03
    var = mrna.loc[train_ids].var(axis=0)
    cutoff = np.percentile(var, cfg["preprocess"]["min_gene_variance_percentile"])
    mrna_f = mrna[var[var >= cutoff].index]

    arm_map = s3.chrom_arm_map(genesets_dir / "hgnc_complete_set.txt")
    arm_burden = s3.cna_arm_burden(cna, arm_map)

    dur_tr = clinical.loc[train_ids, "OS_MONTHS_NUM"].values
    ev_tr = clinical.loc[train_ids, "OS_EVENT"].values
    dur_te = clinical.loc[test_ids, "OS_MONTHS_NUM"].values
    ev_te = clinical.loc[test_ids, "OS_EVENT"].values

    rows = []
    for si, seed in enumerate(args.seeds):
        gene_sets = build_sets_with_seed(cfg, genesets_dir, seed, logger, s3)
        logger.info("Seed %d: %d gene sets", seed, len(gene_sets))
        mrna_path = ssgsea_scores(mrna_f, gene_sets, cfg, logger)
        cna_path = pd.concat([arm_burden, s3.cna_pathway_scores(cna, gene_sets)], axis=1)
        views = {"mrna_pathway": mrna_path, "cna_pathway": cna_path, "mutation": mut}
        order = mcfg["views"]
        scalers = {v: StandardScaler().fit(views[v].loc[train_ids].values) for v in order}
        std_tr = {v: scalers[v].transform(views[v].loc[train_ids].values) for v in order}
        std_te = {v: scalers[v].transform(views[v].loc[test_ids].values) for v in order}

        Z_tr, Z_te = fit_mofa_project(std_tr, std_te, order, mcfg, seed_offset=400 + si)
        k_sel = select_k(Z_tr, dur_tr, ev_tr, ccfg)
        km = KMeans(n_clusters=k_sel, n_init=ccfg["n_init_per_k"],
                    random_state=ccfg["random_seed"]).fit(Z_tr)
        lab_tr, lab_te = km.labels_, km.predict(Z_te)

        d_tr = pd.get_dummies(lab_tr, prefix="c", drop_first=True).astype(float)
        d_te = pd.get_dummies(lab_te, prefix="c", drop_first=True).reindex(
            columns=d_tr.columns, fill_value=0.0).astype(float)
        df = d_tr.copy(); df["duration"], df["event"] = dur_tr, ev_tr
        cph = CoxPHFitter(penalizer=0.1).fit(df, duration_col="duration", event_col="event")
        risk = cph.predict_partial_hazard(d_te).values.ravel()
        c_idx = concordance_index(dur_te, -risk, ev_te)
        lr_p = multivariate_logrank_test(dur_te, lab_te, ev_te).p_value

        rows.append({"seed": seed, "n_gene_sets": len(gene_sets),
                     "n_mrna_features": mrna_path.shape[1], "n_cna_features": cna_path.shape[1],
                     "n_factors": Z_tr.shape[1], "k_selected": k_sel,
                     "c_index_test": c_idx, "logrank_p_test": lr_p,
                     "ari_vs_reference_train": adjusted_rand_score(ref.loc[train_ids], lab_tr)})
        logger.info("Seed %d: k=%d, test C-index %.3f, log-rank p %.4f, ARI to reference %.3f",
                    seed, k_sel, c_idx, lr_p, rows[-1]["ari_vs_reference_train"])
        del mrna_path, cna_path, views, std_tr, std_te, Z_tr, Z_te
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "geneset_seed_sensitivity.csv", index=False)
    logger.info("")
    logger.info("k selected across draws: %s", df.k_selected.tolist())
    logger.info("test C-index: %s (reference 0.561)", [round(x, 3) for x in df.c_index_test])
    logger.info("ARI to the reference partition: %s",
                [round(x, 3) for x in df.ari_vs_reference_train])
    logger.info("Stage 18 complete.")


if __name__ == "__main__":
    main()
