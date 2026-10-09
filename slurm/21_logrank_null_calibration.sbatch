#!/usr/bin/env bash
#SBATCH --job-name=21_logrank_null_calibration
#SBATCH --output=outputs/logs/slurm_21_logrank_null_calibration_%j.out
#SBATCH --error=outputs/logs/slurm_21_logrank_null_calibration_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
source envs/venv/bin/activate 2>/dev/null || conda activate metabric-survival-omics
python scripts/21_logrank_null_calibration.py --config config/config.yaml
