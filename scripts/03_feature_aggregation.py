#!/usr/bin/env python3
"""Stage 03: feature engineering / dimensionality reduction.

Transforms the raw, per-gene omics layers produced by stage 02 into a
compact, biologically interpretable feature set:
  - mRNA  -> single-sample GSEA (ssGSEA) pathway activity scores (Hallmark,
             KEGG, curated Reactome subset)                      -> view "mrna_pathway"
  - CNA   -> chromosome-arm burden scores + pathway-level mean CNA scores
             (same gene sets as the mRNA view)                   -> view "cna_pathway"
  - Mutations -> binarised recurrently-mutated genes + mutational burden
                                                                    -> view "mutation"
  - Clinical variables are carried through separately (not folded into any
    MOFA view), for use as covariates in later stages.

Leakage control: any statistic estimated from data (low-variance gene filter,
choice of the top-N recurrently mutated genes) is fit on the TRAIN split only
(from stage 02's `split` column) and then applied unchanged to the test
patients. ssGSEA itself is a per-sample rank transform and uses no
cross-sample statistics, so it is safe to compute jointly.
"""
from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import pandas as pd
import gseapy as gp

from common import load_config, get_logger, resolve


def read_gmt(path) -> dict[str, list[str]]:
    sets = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            term = parts[0]
            genes = [g for g in parts[2:] if g]
            if genes:
                sets[term] = genes
    return sets


def build_gene_sets(cfg: dict, genesets_dir, logger) -> dict[str, list[str]]:
    ssgsea_cfg = cfg["feature_aggregation"]["ssgsea"]
    min_size, max_size = ssgsea_cfg["min_geneset_size"], ssgsea_cfg["max_geneset_size"]
    rng = np.random.RandomState(cfg["project"]["random_seed"])

    combined = {}
    for gs_cfg in cfg["feature_aggregation"]["genesets"]:
        name, max_terms = gs_cfg["name"], gs_cfg.get("max_terms")
        raw = read_gmt(genesets_dir / f"{name}.gmt")
        filtered = {f"{name.upper()}__{term}": genes
                    for term, genes in raw.items() if min_size <= len(genes) <= max_size}
        if max_terms and len(filtered) > max_terms:
            keys = sorted(filtered.keys())  # deterministic ordering before random choice
            keep = rng.choice(keys, size=max_terms, replace=False)
            filtered = {k: filtered[k] for k in keep}
        logger.info("Gene set '%s': %d raw terms -> %d after size filter/subsample", name, len(raw), len(filtered))
        combined.update(filtered)
    return combined


def chrom_arm_map(hgnc_path) -> pd.Series:
    hgnc = pd.read_csv(hgnc_path, sep="\t", usecols=["symbol", "location"], dtype=str)
    hgnc = hgnc.dropna(subset=["location"])
    arm = hgnc["location"].str.extract(r"^(\d+|X|Y)([pq])")
    valid = arm[0].notna() & arm[1].notna()
    hgnc = hgnc[valid].copy()
    hgnc["arm"] = arm.loc[valid, 0] + arm.loc[valid, 1]
    return hgnc.drop_duplicates(subset="symbol").set_index("symbol")["arm"]


def cna_arm_burden(cna: pd.DataFrame, arm_map: pd.Series) -> pd.DataFrame:
    genes = [g for g in cna.columns if g in arm_map.index]
    sub = cna[genes]
    arms = arm_map.loc[genes]
    out = sub.T.groupby(arms.values).mean().T
    out.columns = [f"ARM__{c}" for c in out.columns]
    return out


def cna_pathway_scores(cna: pd.DataFrame, gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    cols = {}
    cna_genes = set(cna.columns)
    for term, genes in gene_sets.items():
        present = [g for g in genes if g in cna_genes]
        if len(present) >= 5:
            cols[f"CNA_PATH__{term}"] = cna[present].mean(axis=1)
    return pd.DataFrame(cols, index=cna.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("03_feature_aggregation", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    genesets_dir = resolve(cfg["paths"]["genesets_dir"])

    clinical = pd.read_parquet(proc_dir / "clinical.parquet")
    mrna = pd.read_parquet(proc_dir / "mrna_raw.parquet")
    cna = pd.read_parquet(proc_dir / "cna_raw.parquet")
    mut = pd.read_parquet(proc_dir / "mutation_binary.parquet")
    train_ids = clinical.index[clinical["split"] == "train"]

    # ---- low-variance gene filter, fit on train only -----------------------------------
    pcfg = cfg["preprocess"]
    train_var = mrna.loc[train_ids].var(axis=0)
    cutoff = np.percentile(train_var, pcfg["min_gene_variance_percentile"])
    keep_genes = train_var[train_var >= cutoff].index
    mrna_f = mrna[keep_genes]
    logger.info("mRNA genes retained after train-fit variance filter: %d / %d", len(keep_genes), mrna.shape[1])

    # ---- gene sets -----------------------------------------------------------------------
    gene_sets = build_gene_sets(cfg, genesets_dir, logger)
    logger.info("Total combined gene sets for ssGSEA/CNA pathway scoring: %d", len(gene_sets))

    # ---- ssGSEA on mRNA, batched over samples to bound peak memory on an 8GB host --------
    ssgsea_cfg = cfg["feature_aggregation"]["ssgsea"]
    batch_size = ssgsea_cfg.get("sample_batch_size") or mrna_f.shape[0]
    logger.info("Running ssGSEA on %d samples x %d genes x %d gene sets (batch_size=%d) ...",
                mrna_f.shape[0], mrna_f.shape[1], len(gene_sets), batch_size)
    all_ids = list(mrna_f.index)
    batches = [all_ids[i:i + batch_size] for i in range(0, len(all_ids), batch_size)]
    pathway_chunks = []
    for bi, batch_ids in enumerate(batches, 1):
        expr = mrna_f.loc[batch_ids].T
        expr.index.name = "Gene"
        res = gp.ssgsea(
            data=expr, gene_sets=gene_sets, outdir=None,
            sample_norm_method=ssgsea_cfg["sample_norm_method"],
            min_size=ssgsea_cfg["min_geneset_size"], max_size=ssgsea_cfg["max_geneset_size"],
            threads=ssgsea_cfg["threads"], no_plot=True,
        )
        chunk = res.res2d.pivot(index="Name", columns="Term", values="NES").astype(float)
        pathway_chunks.append(chunk)
        del res, expr
        gc.collect()
        logger.info("ssGSEA batch %d/%d complete (%d samples)", bi, len(batches), len(batch_ids))
    mrna_pathway = pd.concat(pathway_chunks, axis=0).reindex(mrna.index)
    logger.info("mRNA pathway score matrix: %s", mrna_pathway.shape)

    # ---- CNA arm-level burden + pathway-level mean scores ---------------------------------
    arm_map = chrom_arm_map(genesets_dir / "hgnc_complete_set.txt")
    arm_burden = cna_arm_burden(cna, arm_map)
    cna_path = cna_pathway_scores(cna, gene_sets)
    cna_pathway = pd.concat([arm_burden, cna_path], axis=1)
    logger.info("CNA view (arm burden + pathway scores): %s", cna_pathway.shape)

    # ---- mutation features: top-N recurrent genes fit on TRAIN, + TMB --------------------
    mcfg = cfg["feature_aggregation"]["mutations"]
    train_recurrence = mut.loc[train_ids].sum(axis=0)
    eligible = train_recurrence[train_recurrence >= cfg["preprocess"]["mutation_min_recurrence"]]
    top_genes = eligible.sort_values(ascending=False).head(mcfg["recurrent_gene_list_top_n"]).index.tolist()
    mutation_feat = mut[top_genes].copy()
    mutation_feat.columns = [f"MUT__{g}" for g in mutation_feat.columns]
    if mcfg["include_tmb"]:
        mutation_feat["TMB"] = mut.sum(axis=1)
    logger.info("Mutation view: %s (top-N genes fit on train, N=%d)", mutation_feat.shape, len(top_genes))

    # ---- clinical covariates (kept separate; encoded for downstream use) -----------------
    cov_cols = cfg["feature_aggregation"]["clinical_covariates"]
    covariates = clinical[cov_cols].copy()
    for c in covariates.columns:
        if covariates[c].dtype == object:
            covariates[c] = covariates[c].replace("", np.nan)
    covariates["AGE_AT_DIAGNOSIS"] = pd.to_numeric(covariates["AGE_AT_DIAGNOSIS"], errors="coerce")
    covariates["GRADE"] = pd.to_numeric(covariates["GRADE"], errors="coerce")
    covariates["TUMOR_SIZE"] = pd.to_numeric(covariates["TUMOR_SIZE"], errors="coerce")
    covariates["TUMOR_STAGE"] = pd.to_numeric(covariates["TUMOR_STAGE"], errors="coerce")
    covariates["LYMPH_NODES_EXAMINED_POSITIVE"] = pd.to_numeric(covariates["LYMPH_NODES_EXAMINED_POSITIVE"], errors="coerce")

    # ---- persist ---------------------------------------------------------------------------
    mrna_pathway.to_parquet(proc_dir / "view_mrna_pathway.parquet")
    cna_pathway.to_parquet(proc_dir / "view_cna_pathway.parquet")
    mutation_feat.to_parquet(proc_dir / "view_mutation.parquet")
    covariates.to_parquet(proc_dir / "clinical_covariates.parquet")

    meta = {
        "n_mrna_genes_kept": int(len(keep_genes)),
        "n_gene_sets_total": len(gene_sets),
        "gene_set_names": sorted(gene_sets.keys()),
        "n_mrna_pathway_features": int(mrna_pathway.shape[1]),
        "n_cna_pathway_features": int(cna_pathway.shape[1]),
        "n_mutation_features": int(mutation_feat.shape[1]),
        "top_mutated_genes": top_genes,
    }
    with open(proc_dir / "feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Stage 03 complete. Feature counts: mrna_pathway=%d, cna_pathway=%d, mutation=%d",
        mrna_pathway.shape[1], cna_pathway.shape[1], mutation_feat.shape[1],
    )


if __name__ == "__main__":
    main()
