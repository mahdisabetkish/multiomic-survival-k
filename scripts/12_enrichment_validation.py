#!/usr/bin/env python3
"""Stage 12: canonical Hallmark signature profile of the clusters (descriptive).

Summarises each cluster's mean activity on a fixed set of canonical Hallmark
signatures (proliferation, ER signalling, PI3K/AKT/mTOR, EMT, DNA repair,
p53, immune response) and records the resulting cluster ranking on each
axis, so that it can be read against each cluster's dominant PAM50-surrogate
composition and against the directions described for the intrinsic
subtypes (Perou et al. 2000; Prat & Perou 2011; TCGA 2012; Curtis et al.
2012).

This stage is descriptive and runs no hypothesis tests. The signature scores
are among the inputs to the factor analysis from which the clusters were
built, so a test of whether they differ between clusters on the same
patients would not be a valid test of anything, and it is not reported.
The output is a table of per-cluster means, the standardised spread across
clusters, and the ranking.

Output: outputs/tables/enrichment_validation.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import load_config, get_logger, resolve

# term -> biological axis
CANONICAL_TERMS = {
    "HALLMARK__E2F Targets": "proliferation",
    "HALLMARK__G2-M Checkpoint": "proliferation",
    "HALLMARK__Myc Targets V1": "proliferation",
    "HALLMARK__Mitotic Spindle": "proliferation",
    "HALLMARK__Estrogen Response Early": "ER_signaling",
    "HALLMARK__Estrogen Response Late": "ER_signaling",
    "HALLMARK__PI3K/AKT/mTOR  Signaling": "PI3K_AKT_mTOR",
    "HALLMARK__Epithelial Mesenchymal Transition": "EMT",
    "HALLMARK__DNA Repair": "DNA_repair",
    "HALLMARK__p53 Pathway": "p53",
    "HALLMARK__Interferon Gamma Response": "immune",
    "HALLMARK__Inflammatory Response": "immune",
}

# composite axes reported alongside the individual terms
COMPOSITES = {
    "proliferation (composite)": ["HALLMARK__E2F Targets", "HALLMARK__G2-M Checkpoint", "HALLMARK__Myc Targets V1"],
    "ER_signaling (composite)": ["HALLMARK__Estrogen Response Early", "HALLMARK__Estrogen Response Late"],
}


def profile_rows(score: pd.Series, labels: pd.Series, clusters, term: str, axis: str) -> dict:
    means = {c: score[labels == c].mean() for c in clusters}
    pooled_sd = score.std(ddof=1)
    # spread of cluster means relative to the overall SD of the score: a descriptive effect size
    spread = (max(means.values()) - min(means.values())) / pooled_sd if pooled_sd > 0 else np.nan
    ranking = ">".join(str(c) for c in sorted(clusters, key=lambda c: -means[c]))
    row = {"term": term, "biological_axis": axis, "cluster_ranking_high_to_low": ranking,
           "range_of_cluster_means_in_sd_units": spread}
    row.update({f"cluster_{c}_mean": means[c] for c in clusters})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("12_enrichment_validation", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])

    mrna = pd.read_parquet(proc_dir / "view_mrna_pathway.parquet")
    cl_train = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]
    mrna = mrna.loc[cl_train.index]
    clusters = sorted(cl_train.unique())

    rows = []
    for term, axis in CANONICAL_TERMS.items():
        if term not in mrna.columns:
            logger.warning("Term not found in mRNA pathway view: %s", term)
            continue
        rows.append(profile_rows(mrna[term], cl_train, clusters, term, axis))
    for name, terms in COMPOSITES.items():
        present = [t for t in terms if t in mrna.columns]
        if present:
            rows.append(profile_rows(mrna[present].mean(axis=1), cl_train, clusters, name, name.split(" ")[0]))

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "enrichment_validation.csv", index=False)
    logger.info("Canonical Hallmark signature profile by cluster (descriptive):\n%s",
                df[["term", "biological_axis", "cluster_ranking_high_to_low",
                    "range_of_cluster_means_in_sd_units"]].to_string(index=False))
    logger.info("Stage 12 complete.")


if __name__ == "__main__":
    main()
