#!/usr/bin/env bash
#SBATCH --job-name=08_external_validation_tcga
#SBATCH --output=outputs/logs/slurm_08_external_validation_tcga_%j.out
#SBATCH --error=outputs/logs/slurm_08_external_validation_tcga_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
source envs/venv/bin/activate 2>/dev/null || conda activate metabric-survival-omics
python scripts/08_external_validation_tcga.py --config config/config.yaml
