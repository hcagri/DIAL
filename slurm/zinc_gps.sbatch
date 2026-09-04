#!/bin/bash

#SBATCH --job-name=dial-zinc-gps
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint='titan_rtx|geforce_rtx_3090|a6000|a100'
#SBATCH --array=0-9
#SBATCH --output=/itet-stor/speltonen/net_scratch/dial/log/%x-%A_%a.out
#SBATCH --error=/itet-stor/speltonen/net_scratch/dial/log/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/itet-stor/speltonen/net_scratch/dial}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/log}"
CONDA_ROOT="${CONDA_ROOT:-/itet-stor/speltonen/net_scratch/conda}"
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-dial}"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENVIRONMENT}"
cd "${PROJECT_ROOT}/zinc"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIGS=(gps.yaml dial_gps.yaml)
SEEDS_PER_CONFIG=5
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as an array job}"
CONFIG_INDEX=$((TASK_ID / SEEDS_PER_CONFIG))
SEED=$((TASK_ID % SEEDS_PER_CONFIG))
CONFIG="${CONFIGS[CONFIG_INDEX]}"
CONFIG_STEM="${CONFIG%.yaml}"
RESULT_DIR="${LOG_ROOT}/results/zinc/${CONFIG_STEM}"

mkdir -p "${RESULT_DIR}"

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Job: ${SLURM_JOB_ID}; array task: ${TASK_ID}"
echo "Experiment: ZINC; config: ${CONFIG}; seed: ${SEED}"
echo "Results: ${RESULT_DIR}"

srun python -u runner.py \
  --config_dirpath ../configs/zinc \
  --config_name "${CONFIG}" \
  --seed "${SEED}" \
  --num_seeds 1 \
  --gpu_id 0 \
  --out_dirpath "${RESULT_DIR}"

echo "Finished: $(date)"
