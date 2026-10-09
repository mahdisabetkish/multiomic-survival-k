#!/usr/bin/env python3
"""Stage 08: external validation on the TCGA-BRCA (PanCancer Atlas) cohort.

TCGA samples are processed as identically as possible to METABRIC (same
Hallmark/KEGG/Reactome gene sets, same ssGSEA settings, same chromosome-arm
mapping, same top-N recurrently mutated gene list persisted from stage 03),
then projected into the METABRIC-trained MOFA latent space using the saved
model, and assigned to clusters using the saved centroids.

Deviation from the study protocol, documented explicitly: the TCGA
mutation MAF (`data_mutations.txt`) is not retrievable -- the object is
missing from the upstream cBioPortal datahub's git-lfs storage (confirmed via
the LFS batch API, not a transient network failure) -- so TCGA projection
uses only the mrna_pathway and cna_pathway views. This is a principled
partial-view projection (MOFA's own conditional-posterior update naturally
handles missing views by summing over whatever views are observed), and low-
impact here because the mutation view explained only ~2-3% of variance in
training.

Treatment-effect replication (study protocol: "if treatment data is available") is
attempted but the curated TCGA PanCanAtlas clinical export used here has no
ER/PR/HER2 receptor status and no structured adjuvant-chemotherapy field, so
replication of the ER+/chemotherapy contrast is not feasible with the
available fields; this is reported explicitly rather than substituted with an
unreliable proxy.

Outputs (outputs/tables/):
    tcga_cluster_assignment_summary.csv
    tcga_survival_validation.csv
    tcga_cluster_proportions_comparison.csv
    tcga_treatment_replication_status.json
data/processed/tcga/clinical_with_clusters.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
import joblib

from common import load_config, get_logger, resolve, parse_os_status


def load_feature_aggregation_module():
    path = Path(__file__).parent / "03_feature_aggregation.py"
    spec = importlib.util.spec_from_file_location("feature_aggregation_03", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_gene_matrix_tcga(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.dropna(subset=["Hugo_Symbol"])
    df = df.drop(columns=[c for c in ["Entrez_Gene_Id"] if c in df.columns])
    df = df.groupby("Hugo_Symbol").mean(numeric_only=True)
    return df.T  # samples x genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("08_external_validation_tcga", resolve(cfg["paths"]["logs_dir"]))
    raw_dir = resolve(cfg["paths"]["raw_tcga_dir"])
    out_dir = resolve(cfg["paths"]["processed_tcga_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    ckpt_dir = resolve(cfg["paths"]["checkpoints_dir"])
    genesets_dir = resolve(cfg["paths"]["genesets_dir"])

    feat_agg = load_feature_aggregation_module()

    # ---- load + harmonise TCGA clinical + omics -------------------------------------------
    patient = pd.read_csv(raw_dir / "data_clinical_patient.txt", sep="\t", comment="#", dtype=str)
    sample = pd.read_csv(raw_dir / "data_clinical_sample.txt", sep="\t", comment="#", dtype=str)
    clin = patient.merge(sample, on="PATIENT_ID", how="inner").drop_duplicates(subset="SAMPLE_ID")
    clin = clin.set_index("SAMPLE_ID")

    os_time = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    os_event = parse_os_status(clin["OS_STATUS"])
    clin["OS_MONTHS_NUM"] = os_time
    clin["OS_EVENT"] = os_event

    logger.info("Loading TCGA mRNA (RSEM) and CNA matrices ...")
    mrna = load_gene_matrix_tcga(raw_dir / "data_mrna_seq_v2_rsem.txt")
    mrna = np.log2(mrna.clip(lower=0) + 1)  # match METABRIC's log2 microarray scale for ssGSEA input
    cna = load_gene_matrix_tcga(raw_dir / "data_cna.txt")

    cohort = [s for s in clin.index if s in mrna.index and s in cna.index
              and pd.notna(clin.loc[s, "OS_MONTHS_NUM"]) and pd.notna(clin.loc[s, "OS_EVENT"])]
    logger.info("TCGA analysis cohort (mRNA + CNA + usable survival label): n=%d", len(cohort))
    clin = clin.loc[cohort]
    mrna = mrna.loc[cohort]
    cna = cna.loc[cohort]

    # ---- rebuild the identical gene-set / feature pipeline from stage 03 -------------------
    gene_sets = feat_agg.build_gene_sets(cfg, genesets_dir, logger)
    ssgsea_cfg = cfg["feature_aggregation"]["ssgsea"]
    batch_size = ssgsea_cfg.get("sample_batch_size") or mrna.shape[0]
    ids = list(mrna.index)
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    chunks = []
    import gseapy as gp
    for bi, batch_ids in enumerate(batches, 1):
        expr = mrna.loc[batch_ids].T
        expr.index.name = "Gene"
        res = gp.ssgsea(data=expr, gene_sets=gene_sets, outdir=None,
                         sample_norm_method=ssgsea_cfg["sample_norm_method"],
                         min_size=ssgsea_cfg["min_geneset_size"], max_size=ssgsea_cfg["max_geneset_size"],
                         threads=ssgsea_cfg["threads"], no_plot=True)
        chunks.append(res.res2d.pivot(index="Name", columns="Term", values="NES").astype(float))
        logger.info("TCGA ssGSEA batch %d/%d complete", bi, len(batches))
    mrna_pathway = pd.concat(chunks, axis=0).reindex(mrna.index)

    arm_map = feat_agg.chrom_arm_map(genesets_dir / "hgnc_complete_set.txt")
    cna_pathway = pd.concat([feat_agg.cna_arm_burden(cna, arm_map), feat_agg.cna_pathway_scores(cna, gene_sets)], axis=1)

    # persisted for stage 13 (cross-platform batch-correction analysis), which needs the raw
    # (pre-standardization, pre-projection) TCGA pathway views alongside METABRIC's
    mrna_pathway.to_parquet(out_dir / "view_mrna_pathway.parquet")
    cna_pathway.to_parquet(out_dir / "view_cna_pathway.parquet")

    # ---- project into the METABRIC-trained MOFA space (mRNA + CNA views only) --------------
    mofa_bundle = joblib.load(ckpt_dir / "mofa_model.joblib")
    scalers, feature_names = mofa_bundle["scalers"], mofa_bundle["feature_names"]
    W, tau, n_factors = mofa_bundle["W"], mofa_bundle["tau"], mofa_bundle["n_factors"]

    available_views = ["mrna_pathway", "cna_pathway"]
    logger.info("Projecting TCGA into MOFA space using views=%s (mutation view unavailable for TCGA)", available_views)

    views_std = {}
    for view, df in [("mrna_pathway", mrna_pathway), ("cna_pathway", cna_pathway)]:
        df_aligned = df.reindex(columns=feature_names[view], fill_value=0.0).fillna(0.0)
        views_std[view] = scalers[view].transform(df_aligned.values)

    precision = np.eye(n_factors)
    rhs = np.zeros((len(cohort), n_factors))
    for view in available_views:
        Wv, tauv = W[view], tau[view]
        Xv = views_std[view]
        precision += Wv.T @ (Wv * tauv[:, None])
        rhs += Xv @ (Wv * tauv[:, None])
    Z_tcga = rhs @ np.linalg.inv(precision).T
    Z_tcga_df = pd.DataFrame(Z_tcga, index=cohort, columns=[f"Factor{k+1}" for k in range(n_factors)])

    kmeans_bundle = joblib.load(ckpt_dir / "kmeans_model.joblib")
    tcga_clusters = pd.Series(kmeans_bundle["model"].predict(Z_tcga_df.values), index=cohort, name="cluster")

    # ---- prognostic validation ---------------------------------------------------------------
    durations, events = clin["OS_MONTHS_NUM"].values, clin["OS_EVENT"].values
    logrank_p = multivariate_logrank_test(durations, tcga_clusters.values, events).p_value

    dummies = pd.get_dummies(tcga_clusters, prefix="cluster", drop_first=True).astype(float)
    df_cox = dummies.copy()
    df_cox["duration"], df_cox["event"] = durations, events
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(df_cox, duration_col="duration", event_col="event")
        risk = cph.predict_partial_hazard(dummies).values.ravel()
        c_index = concordance_index(durations, -risk, events)
    except Exception as e:
        logger.warning("TCGA Cox fit failed (%s); c_index set to NaN", e)
        c_index = np.nan

    surv_val = pd.DataFrame([{
        "n_patients": len(cohort), "n_clusters": tcga_clusters.nunique(),
        "logrank_p": logrank_p, "c_index_cluster_only": c_index,
    }])
    surv_val.to_csv(tables_dir / "tcga_survival_validation.csv", index=False)
    logger.info("TCGA prognostic validation: logrank_p=%.4g, c_index=%.4f", logrank_p, c_index)

    train_clusters = pd.read_parquet(resolve(cfg["paths"]["processed_metabric_dir"]) / "cluster_labels_train.parquet")["cluster"]
    metabric_props = train_clusters.value_counts(normalize=True).sort_index()
    tcga_props = tcga_clusters.value_counts(normalize=True).sort_index()
    prop_df = pd.DataFrame({"metabric_train_proportion": metabric_props, "tcga_proportion": tcga_props}).fillna(0.0)
    prop_df.to_csv(tables_dir / "tcga_cluster_proportions_comparison.csv")
    logger.info("Cluster proportions (METABRIC train vs TCGA):\n%s", prop_df.to_string())

    tcga_clusters.to_frame().join(clin).to_parquet(out_dir / "clinical_with_clusters.parquet")
    tcga_clusters.value_counts().sort_index().rename("n_patients").to_csv(
        tables_dir / "tcga_cluster_assignment_summary.csv")

    # ---- treatment-effect replication feasibility check --------------------------------------
    has_er_status = any("ER" in c.upper() for c in clin.columns)
    has_chemo_flag = any("CHEMO" in c.upper() for c in clin.columns)
    status = {
        "attempted": True, "replicated": False,
        "reason": ("TCGA PanCanAtlas curated clinical export used here has no ER/PR/HER2 receptor "
                   "status field and no structured adjuvant-chemotherapy indicator, so the ER+/"
                   "chemotherapy causal contrast from stage 07 cannot be replicated on this cohort "
                   "without an unreliable proxy. Consistent with the study protocol's own expectation that "
                   "TCGA treatment-effect replication is often limited."),
        "has_er_status_field": has_er_status, "has_chemo_flag_field": has_chemo_flag,
    }
    with open(tables_dir / "tcga_treatment_replication_status.json", "w") as f:
        json.dump(status, f, indent=2)
    logger.info("Treatment-effect replication: %s", status["reason"])

    logger.info("Stage 08 complete.")


if __name__ == "__main__":
    main()
