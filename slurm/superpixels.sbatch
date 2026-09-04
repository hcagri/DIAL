#!/bin/bash

#SBATCH --job-name=dial-cifar10
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=/itet-stor/speltonen/net_scratch/dial/log/%x-%A_%a.out
#SBATCH --error=/itet-stor/speltonen/net_scratch/dial/log/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/itet-stor/speltonen/net_scratch/dial}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/log}"
CONDA_ROOT="${CONDA_ROOT:-/itet-stor/speltonen/net_scratch/conda}"
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-dial}"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENVIRONMENT}"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIGS=(baseline rpearl dial)
SEEDS_PER_CONFIG=3
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as an array job}"
CONFIG_INDEX=$((TASK_ID / SEEDS_PER_CONFIG))
SEED=$((TASK_ID % SEEDS_PER_CONFIG))
CONFIG="${CONFIGS[CONFIG_INDEX]}"
RESULT_DIR="${LOG_ROOT}/results/superpixels/${CONFIG}"

mkdir -p "${RESULT_DIR}"

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Job: ${SLURM_JOB_ID}; array task: ${TASK_ID}"
echo "Experiment: CIFAR10 superpixels; config: ${CONFIG}; seed: ${SEED}"
echo "Results: ${RESULT_DIR}"

srun python -u -m superpixels.runner \
  --config_name "${CONFIG}" \
  --dataset_name CIFAR10 \
  --seed "${SEED}" \
  --num_seeds 1 \
  --gpu_id 0 \
  --out_dirpath "${RESULT_DIR}"

echo "Finished: $(date)"
