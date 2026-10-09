#!/usr/bin/env bash
#SBATCH --job-name=15_overlap_weighting
#SBATCH --output=outputs/logs/slurm_15_overlap_weighting_%j.out
#SBATCH --error=outputs/logs/slurm_15_overlap_weighting_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
source envs/venv/bin/activate 2>/dev/null || conda activate metabric-survival-omics
python scripts/15_overlap_weighting.py --config config/config.yaml
