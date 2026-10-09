#!/usr/bin/env bash
#SBATCH --job-name=19_positive_control_k_range
#SBATCH --output=outputs/logs/slurm_19_positive_control_k_range_%j.out
#SBATCH --error=outputs/logs/slurm_19_positive_control_k_range_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
source envs/venv/bin/activate 2>/dev/null || conda activate metabric-survival-omics
python scripts/19_positive_control_k_range.py --config config/config.yaml
