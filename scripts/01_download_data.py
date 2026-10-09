#!/usr/bin/env python3
"""Stage 01: download the public input data.

METABRIC and TCGA-BRCA (PanCancer Atlas) come from the cBioPortal datahub, and the
Hallmark, KEGG and Reactome gene-set libraries from Enrichr. The HGNC gene table is
shipped in data/reference/ (the version used for the paper, retrieved on 7 August 2026),
because HGNC publishes a new table every month and the chromosome-arm copy-number
features depend on it.

Every cBioPortal file is checked against the SHA-256 checksum of the file used for the
paper. A mismatch means the upstream file has changed since then, so results may differ;
the script warns but does not stop. The Enrichr libraries are not byte-identical to the
copies used for the paper, but they contain the same gene sets once the size filter of
stage 03 has been applied.

Usage: python scripts/01_download_data.py --config config/config.yaml [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from common import load_config, resolve

METABRIC_URL = "https://datahub.assets.cbioportal.org/brca_metabric.tar.gz"
METABRIC_SHA256 = "6d4683477d6b37a2d7edbedc0df610f67bc456f99e5e1bef6219f37b633a55f7"
TCGA_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_tcga_pan_can_atlas_2018/"
TCGA_FILES = {
    "data_clinical_patient.txt": "13c939025794ce21905ccb3409865fa08506b2283b432f8e3a3022f60a4c7682",
    "data_clinical_sample.txt": "e96e6e4895a6ddcdb31d508a9e8f5676ca78ce396d2f82cd111a286a17e75ca6",
    "data_cna.txt": "4b5a97e88b6358fcdec8422e3c4816c6626c2b317ef9053e6d20d131c695c1b8",
    "data_mrna_seq_v2_rsem.txt": "aba030e1f04a196ee6c2118ed37e777b533f7fd24ffd1bcbde908dcee33b2eb5",
}
ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={}"
HGNC_FILE = "hgnc_complete_set.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        done, report = 0, 100 << 20
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            done += len(block)
            if done >= report:
                print(f"  {dest.name}: {done >> 20} MB")
                report += 100 << 20
    tmp.replace(dest)


def check(path: Path, expected: str) -> None:
    got = sha256(path)
    status = "matches the file used for the paper" if got == expected else \
        "WARNING: differs from the file used for the paper; results may differ"
    print(f"  {path.name}: {status}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--force", action="store_true", help="download again even if the files exist")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    metabric_dir, tcga_dir = resolve(paths["raw_metabric_dir"]), resolve(paths["raw_tcga_dir"])
    genesets_dir = resolve(paths["genesets_dir"])

    print("METABRIC (cBioPortal, brca_metabric)")
    if args.force or not (metabric_dir / "data_mrna_illumina_microarray.txt").exists():
        archive = metabric_dir.parent / "brca_metabric.tar.gz"
        download(METABRIC_URL, archive)
        check(archive, METABRIC_SHA256)
        metabric_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue
                member.name = str(Path(*parts[1:]))       # drop the leading brca_metabric/
                tar.extract(member, metabric_dir, filter="data")
        archive.unlink()
    else:
        print("  already present")

    print("TCGA-BRCA (cBioPortal, brca_tcga_pan_can_atlas_2018)")
    for name, expected in TCGA_FILES.items():
        dest = tcga_dir / name
        if args.force or not dest.exists():
            download(TCGA_BASE + name, dest)
        check(dest, expected)

    print("Gene sets (Enrichr) and HGNC table")
    for gs in cfg["feature_aggregation"]["genesets"]:
        dest = genesets_dir / f"{gs['name']}.gmt"
        if args.force or not dest.exists():
            download(ENRICHR_URL.format(gs["source"]), dest)
        print(f"  {dest.name}: {gs['source']}")
    shutil.copy(resolve("data/reference") / HGNC_FILE, genesets_dir / HGNC_FILE)
    print(f"  {HGNC_FILE}: copied from data/reference/")
    print("Done.")


if __name__ == "__main__":
    main()
