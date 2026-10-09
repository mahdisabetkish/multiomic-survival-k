#!/usr/bin/env bash
#SBATCH --job-name=13_cross_platform_correction
#SBATCH --output=outputs/logs/slurm_13_cross_platform_correction_%j.out
#SBATCH --error=outputs/logs/slurm_13_cross_platform_correction_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
source envs/venv/bin/activate 2>/dev/null || conda activate metabric-survival-omics
python scripts/13_cross_platform_correction.py --config config/config.yaml
