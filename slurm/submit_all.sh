#!/usr/bin/env bash
# Master submission script: chains all pipeline stages as dependent SLURM jobs
# (stage N+1 starts only after stage N exits successfully), so the pipeline can
# be launched on a CPU-only HPC cluster with a single command:
#   bash slurm/submit_all.sh
# Run scripts/01_download_data.py first; it needs internet access. The results
# in outputs/ were produced by running the same stages locally, without SLURM.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGES=(
  02_preprocess_omics
  03_feature_aggregation
  04_mofa_factorisation
  05_survival_aware_clustering
  06_cluster_evaluation
  07_causal_treatment_effects
  08_external_validation_tcga
  10_baseline_comparison
  11_ablation_study
  12_enrichment_validation
  13_cross_platform_correction
  07b_nested_oof_sensitivity
  14_full_pipeline_stability
  15_overlap_weighting
  16_positive_control
  17_inner_repeat_sensitivity
  18_geneset_seed_sensitivity
  19_positive_control_k_range
  20_repeated_outer_splits
  21_logrank_null_calibration
  22_fixed_k_control
  09_generate_figures_tables
)

prev_jobid=""
for stage in "${STAGES[@]}"; do
  if [ -z "$prev_jobid" ]; then
    dep_args=()
  else
    dep_args=(--dependency=afterok:"$prev_jobid")
  fi
  jobid=$(sbatch --parsable "${dep_args[@]}" "slurm/${stage}.sbatch")
  echo "submitted ${stage} as job ${jobid}${prev_jobid:+ (after ${prev_jobid})}"
  prev_jobid="$jobid"
done
