#!/usr/bin/env python3
"""Stage 02: load raw cBioPortal METABRIC files, harmonise patient/sample IDs
across mRNA, CNA, mutation and clinical layers, define the analysis cohort,
and perform the leakage-free stratified train/test split.

Per the study protocol, the split happens before any filtering, scaling, or imputation
that uses data-driven statistics. This script only does structural
harmonisation (ID alignment, dropping patients with missing survival data) --
no gene filtering, no scaling -- so it is safe to run before the split, and
the split itself is produced here so every downstream stage can condition on
it.

Outputs (data/processed/metabric/):
    clinical.parquet        patient-level clinical + split column
    mrna_raw.parquet        samples x genes, raw log2 microarray intensities
    cna_raw.parquet         samples x genes, GISTIC-style calls in {-2,-1,0,1,2}
    mutation_binary.parquet patients x genes, 1 = functional mutation present
    train_patient_ids.txt
    test_patient_ids.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import load_config, get_logger, parse_os_status, parse_rfs_status, resolve

FUNCTIONAL_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Translation_Start_Site",
}


def load_clinical(raw_dir: Path) -> pd.DataFrame:
    patient = pd.read_csv(raw_dir / "data_clinical_patient.txt", sep="\t", comment="#", dtype=str)
    sample = pd.read_csv(raw_dir / "data_clinical_sample.txt", sep="\t", comment="#", dtype=str)
    # one sample per patient in METABRIC; keep first if duplicated
    sample = sample.drop_duplicates(subset="PATIENT_ID", keep="first")
    clin = patient.merge(sample, on="PATIENT_ID", how="inner", suffixes=("", "_sample"))
    clin = clin.set_index("PATIENT_ID")
    return clin


def load_gene_matrix(path: Path, id_cols=("Hugo_Symbol", "Entrez_Gene_Id")) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.dropna(subset=["Hugo_Symbol"])
    df = df.drop(columns=[c for c in id_cols if c in df.columns and c != "Hugo_Symbol"])
    # collapse duplicate gene symbols (multiple probes/entries) by mean
    df = df.groupby("Hugo_Symbol").mean(numeric_only=True)
    return df.T  # samples x genes


def build_mutation_binary(raw_dir: Path, patient_ids: list[str]) -> pd.DataFrame:
    maf = pd.read_csv(
        raw_dir / "data_mutations.txt", sep="\t", comment="#", low_memory=False,
        usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"],
    )
    maf = maf[maf["Variant_Classification"].isin(FUNCTIONAL_CLASSES)]
    maf = maf[maf["Tumor_Sample_Barcode"].isin(patient_ids)]
    mat = (
        maf.drop_duplicates(subset=["Tumor_Sample_Barcode", "Hugo_Symbol"])
        .assign(mutated=1)
        .pivot(index="Tumor_Sample_Barcode", columns="Hugo_Symbol", values="mutated")
    )
    mat = mat.reindex(index=patient_ids).fillna(0).astype(int)
    mat.index.name = "PATIENT_ID"
    return mat


def make_stratified_split(clin: pd.DataFrame, cfg: dict) -> pd.Series:
    strat_cfg = cfg["split"]
    event = parse_os_status(clin["OS_STATUS"])
    subtype = clin["CLAUDIN_SUBTYPE"].fillna("Unknown").replace("", "Unknown")

    strat_label = event.astype(str) + "_" + subtype.astype(str)
    counts = strat_label.value_counts()
    rare = counts[counts < 4].index
    strat_label = strat_label.where(~strat_label.isin(rare), other=event.astype(str) + "_RARE")

    train_idx, test_idx = train_test_split(
        clin.index,
        test_size=strat_cfg["test_fraction"],
        stratify=strat_label,
        random_state=strat_cfg["random_seed"],
    )
    split = pd.Series("train", index=clin.index, name="split")
    split.loc[test_idx] = "test"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("02_preprocess_omics", resolve(cfg["paths"]["logs_dir"]))
    raw_dir = resolve(cfg["paths"]["raw_metabric_dir"])
    out_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading clinical data from %s", raw_dir)
    clin = load_clinical(raw_dir)
    logger.info("Loaded clinical for %d patients", len(clin))

    logger.info("Loading mRNA microarray matrix")
    mrna = load_gene_matrix(raw_dir / "data_mrna_illumina_microarray.txt")
    logger.info("mRNA matrix: %d samples x %d genes", *mrna.shape)

    logger.info("Loading CNA matrix")
    cna = load_gene_matrix(raw_dir / "data_cna.txt")
    logger.info("CNA matrix: %d samples x %d genes", *cna.shape)

    # analysis cohort: complete omics coverage + usable survival label + PAM50 surrogate
    with open(raw_dir / "case_lists" / "cases_complete.txt") as f:
        for line in f:
            if line.startswith("case_list_ids:"):
                complete_ids = line.split(":", 1)[1].split()
                break
    complete_ids = [i for i in complete_ids if i in clin.index and i in mrna.index and i in cna.index]

    os_time = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    os_event = parse_os_status(clin["OS_STATUS"])
    has_survival = os_time.notna() & os_event.notna()
    has_subtype = clin["CLAUDIN_SUBTYPE"].notna() & (clin["CLAUDIN_SUBTYPE"] != "") & (clin["CLAUDIN_SUBTYPE"] != "NC")

    cohort = [pid for pid in complete_ids if has_survival.get(pid, False) and has_subtype.get(pid, False)]
    logger.info("Analysis cohort after QC: %d patients", len(cohort))

    clin = clin.loc[cohort].copy()
    mrna = mrna.loc[cohort]
    cna = cna.loc[cohort]

    logger.info("Building functional mutation binary matrix")
    mut_bin = build_mutation_binary(raw_dir, cohort)
    logger.info("Mutation matrix: %d patients x %d genes with >=1 mutation", *mut_bin.shape)

    clin["OS_MONTHS_NUM"] = os_time.loc[cohort]
    clin["OS_EVENT"] = os_event.loc[cohort].astype(int)
    clin["RFS_MONTHS_NUM"] = pd.to_numeric(clin["RFS_MONTHS"], errors="coerce")
    clin["RFS_EVENT"] = parse_rfs_status(clin["RFS_STATUS"])
    clin["FUNCTIONAL_MUTATION_COUNT"] = mut_bin.sum(axis=1).reindex(clin.index)

    logger.info("Performing leakage-free stratified train/test split (test_fraction=%.2f)",
                cfg["split"]["test_fraction"])
    split = make_stratified_split(clin, cfg)
    clin["split"] = split
    logger.info("Split sizes: train=%d, test=%d", (split == "train").sum(), (split == "test").sum())

    clin.to_parquet(out_dir / "clinical.parquet")
    mrna.to_parquet(out_dir / "mrna_raw.parquet")
    cna.to_parquet(out_dir / "cna_raw.parquet")
    mut_bin.to_parquet(out_dir / "mutation_binary.parquet")

    (out_dir / "train_patient_ids.txt").write_text("\n".join(split[split == "train"].index) + "\n")
    (out_dir / "test_patient_ids.txt").write_text("\n".join(split[split == "test"].index) + "\n")

    logger.info("Stage 02 complete. Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
