#!/bin/bash
#===============================================================================
# SLURM Job Array: NIG Hyperparameter Search (1 config per job)
# Purpose: Run hyperparameter configurations in parallel across multiple nodes
#
# Usage:
#   1. Generate configs:    python hyperparam_configs.py
#   2. Submit all:          sbatch --array=0-47 slurm_array.sh
#   3. Submit subset:       sbatch --array=0-15 slurm_array.sh
#   4. Limit concurrent:    sbatch --array=0-47%8 slurm_array.sh
#
# Note: Experiment names come from configs file, so resume is automatic
#       if a job is resubmitted - it will continue from checkpoint.
#===============================================================================

#SBATCH --job-name=nig_hp_%a
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/nig_hp_%A_%a.out
#SBATCH --error=logs/slurm/nig_hp_%A_%a.err

#===============================================================================
# Environment Setup
#===============================================================================
PROJECT_DIR="${GRAPHPERF_RT_ROOT}"
NIG_DIR="${PROJECT_DIR}/nig_experiment"
PYTHON="/projects/bdau/envs/cfg_env/bin/python"
CONFIG_FILE="${NIG_DIR}/hyperparam_search/configs_latest.json"

cd "${NIG_DIR}"
mkdir -p logs/slurm hyperparam_search/results

TASK_ID=${SLURM_ARRAY_TASK_ID}

#===============================================================================
# Load Configuration
#===============================================================================
CONFIG=$($PYTHON -c "
import json, sys
with open('${CONFIG_FILE}') as f:
    data = json.load(f)
if ${TASK_ID} >= len(data['configs']):
    print('ERROR', file=sys.stderr)
    sys.exit(1)
c = data['configs'][${TASK_ID}]
print(f\"{c['experiment_name']}|{c['lambda_mse']}|{c['lambda_ns']}|{c['lr']}|{c['hidden_dim']}|{c['epochs']}|{c['batch_size']}\")
")

if [ $? -ne 0 ]; then
    echo "ERROR: Task ID ${TASK_ID} out of range"
    exit 1
fi

IFS='|' read -r EXPERIMENT_NAME LAMBDA_MSE LAMBDA_NS LR HIDDEN_DIM EPOCHS BATCH_SIZE <<< "$CONFIG"

#===============================================================================
# Print Info
#===============================================================================
echo "=============================================="
echo "NIG Hyperparameter Search - Task ${TASK_ID}"
echo "=============================================="
echo "Experiment:  ${EXPERIMENT_NAME}"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Array ID:    ${SLURM_ARRAY_JOB_ID}"
echo "Node:        ${SLURM_NODELIST}"
echo "Start:       $(date)"
echo ""
echo "Config: λ_mse=${LAMBDA_MSE}, λ_ns=${LAMBDA_NS}, lr=${LR}, hidden=${HIDDEN_DIM}"
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
# Save Results Summary
#===============================================================================
RESULTS_FILE="${NIG_DIR}/experiments/${EXPERIMENT_NAME}/results.json"
SUMMARY_FILE="${NIG_DIR}/hyperparam_search/results/task_${TASK_ID}.json"

if [ -f "${RESULTS_FILE}" ]; then
    $PYTHON -c "
import json
with open('${RESULTS_FILE}') as f:
    r = json.load(f)
r['task_id'] = ${TASK_ID}
r['experiment_name'] = '${EXPERIMENT_NAME}'
r['slurm_job_id'] = '${SLURM_JOB_ID}'
with open('${SUMMARY_FILE}', 'w') as f:
    json.dump(r, f, indent=2)
print(f'Task ${TASK_ID}: R²={r.get(\"test_r2\", \"N/A\"):.4f}')
"
else
    echo "{\"task_id\": ${TASK_ID}, \"status\": \"failed\", \"exit_code\": ${EXIT_CODE}}" > "${SUMMARY_FILE}"
    echo "Task ${TASK_ID}: FAILED (exit code: ${EXIT_CODE})"
fi

echo ""
echo "=============================================="
echo "Task ${TASK_ID} completed at $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
