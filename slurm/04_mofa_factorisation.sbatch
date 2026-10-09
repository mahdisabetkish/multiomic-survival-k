#!/usr/bin/env bash
#SBATCH --job-name=04_mofa_factorisation
#SBATCH --output=outputs/logs/slurm_04_mofa_factorisation_%j.out
#SBATCH --error=outputs/logs/slurm_04_mofa_factorisation_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
cd "$(dirname "$0")/.."
source envs/venv/bin/activate 2>/dev/null || conda activate metabric-survival-omics
python scripts/04_mofa_factorisation.py --config config/config.yaml
