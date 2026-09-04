#!/bin/bash

#SBATCH --job-name=dial-cycle-gps
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-23
#SBATCH --output=/itet-stor/speltonen/net_scratch/dial/log/%x-%A_%a.out
#SBATCH --error=/itet-stor/speltonen/net_scratch/dial/log/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/itet-stor/speltonen/net_scratch/dial}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/log}"
CONDA_ROOT="${CONDA_ROOT:-/itet-stor/speltonen/net_scratch/conda}"
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-dial}"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENVIRONMENT}"
cd "${PROJECT_ROOT}/cycle_count_i2gnn"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIGS=(gps.yaml dial_gps.yaml)
CYCLES=(3 4 5 6)
SEEDS_PER_CYCLE=3
TASKS_PER_CONFIG=$((${#CYCLES[@]} * SEEDS_PER_CYCLE))
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as an array job}"
CONFIG_INDEX=$((TASK_ID / TASKS_PER_CONFIG))
WITHIN_CONFIG=$((TASK_ID % TASKS_PER_CONFIG))
CYCLE_INDEX=$((WITHIN_CONFIG / SEEDS_PER_CYCLE))
SEED=$((WITHIN_CONFIG % SEEDS_PER_CYCLE))
CONFIG="${CONFIGS[CONFIG_INDEX]}"
CYCLE="${CYCLES[CYCLE_INDEX]}"
CONFIG_STEM="${CONFIG%.yaml}"
RESULT_DIR="${LOG_ROOT}/results/cycle_counts/${CONFIG_STEM}"

mkdir -p "${RESULT_DIR}"

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Job: ${SLURM_JOB_ID}; array task: ${TASK_ID}"
echo "Experiment: cycle counting; config: ${CONFIG}; cycle: ${CYCLE}; seed: ${SEED}"
echo "Results: ${RESULT_DIR}"

srun python -u runner.py \
  --config_dirpath ../configs/cycle_counts_i2gnn \
  --config_name "${CONFIG}" \
  --cycles "${CYCLE}" \
  --seed "${SEED}" \
  --num_seeds 1 \
  --gpu_id 0 \
  --out_dirpath "${RESULT_DIR}"

echo "Finished: $(date)"
