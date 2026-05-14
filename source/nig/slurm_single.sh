#!/bin/bash
#===============================================================================
# SLURM Job Script: NIG Evidential Training (Single Run)
# Purpose: Train GAT Evidential model with NIG uncertainty for IJCAI paper
#
# Usage:
#   Fresh start:  sbatch slurm_single.sh
#   Resume:       RESUME=nig_ijcai_paper_20260119_001844 sbatch slurm_single.sh
#===============================================================================

#SBATCH --job-name=nig_train
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/nig_train_%j.out
#SBATCH --error=logs/slurm/nig_train_%j.err

#===============================================================================
# Environment Setup
#===============================================================================
PROJECT_DIR="${GRAPHPERF_RT_ROOT}"
NIG_DIR="${PROJECT_DIR}/nig_experiment"
PYTHON="/projects/bdau/envs/cfg_env/bin/python"

cd "${NIG_DIR}"
mkdir -p logs/slurm

#===============================================================================
# Resume or Fresh Start
#===============================================================================
# RESUME env var: set to experiment name to resume, empty for fresh start
if [ -n "${RESUME}" ]; then
    EXPERIMENT_NAME="${RESUME}"
    MODE="RESUME"
else
    EXPERIMENT_NAME="nig_$(date +%Y%m%d_%H%M%S)"
    MODE="FRESH"
fi

#===============================================================================
# Hyperparameters
#===============================================================================
EPOCHS=100
BATCH_SIZE=64
LR=0.001
HIDDEN_DIM=128
LAMBDA_MSE=10.0
LAMBDA_NS=0.01

#===============================================================================
# Print Info
#===============================================================================
echo "=============================================="
echo "NIG Evidential Training"
echo "=============================================="
echo "Mode:        ${MODE}"
echo "Experiment:  ${EXPERIMENT_NAME}"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURM_NODELIST}"
echo "Start:       $(date)"
echo ""
echo "Hyperparameters:"
echo "  epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LR}"
echo "  hidden=${HIDDEN_DIM}, λ_mse=${LAMBDA_MSE}, λ_ns=${LAMBDA_NS}"
echo "=============================================="

#===============================================================================
# Training
#===============================================================================
$PYTHON train.py \
    --experiment "${EXPERIMENT_NAME}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --hidden_dim ${HIDDEN_DIM} \
    --lambda_mse ${LAMBDA_MSE} \
    --lambda_ns ${LAMBDA_NS} \
    --device cuda

EXIT_CODE=$?

#===============================================================================
# Summary
#===============================================================================
echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time:  $(date)"

RESULTS_FILE="${NIG_DIR}/experiments/${EXPERIMENT_NAME}/results.json"
if [ -f "${RESULTS_FILE}" ]; then
    echo ""
    echo "Results:"
    cat "${RESULTS_FILE}"
    echo ""
    echo "Output: ${NIG_DIR}/experiments/${EXPERIMENT_NAME}/"
else
    echo "WARNING: Results not found. Check logs."
fi

exit ${EXIT_CODE}
