#!/usr/bin/env bash
#SBATCH --job-name=17_inner_repeat_sensitivity
#SBATCH --output=outputs/logs/slurm_17_inner_repeat_sensitivity_%j.out
#SBATCH --error=outputs/logs/slurm_17_inner_repeat_sensitivity_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
source envs/venv/bin/activate 2>/dev/null || conda activate metabric-survival-omics
python scripts/17_inner_repeat_sensitivity.py --config config/config.yaml
