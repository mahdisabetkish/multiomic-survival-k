#!/usr/bin/env bash
#SBATCH --job-name=20_repeated_outer_splits
#SBATCH --output=outputs/logs/slurm_20_repeated_outer_splits_%j.out
#SBATCH --error=outputs/logs/slurm_20_repeated_outer_splits_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
source envs/venv/bin/activate 2>/dev/null || conda activate metabric-survival-omics
python scripts/20_repeated_outer_splits.py --config config/config.yaml
