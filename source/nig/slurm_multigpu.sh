#!/bin/bash
#===============================================================================
# SLURM Job: Multi-GPU NIG Hyperparameter Search (8 configs per node)
# Purpose: Run 8 configs in parallel on one node (1 per GPU)
#
# Usage:
#   1. Generate configs:    python hyperparam_configs.py
#   2. Run all 48 configs:  sbatch --array=0-5 slurm_multigpu.sh
#   3. Run first 8 only:    sbatch slurm_multigpu.sh
#
# Each array task runs 8 configs (one per GPU):
#   array=0: configs 0-7,  array=1: configs 8-15, etc.
#
# Note: Resume is automatic - resubmitting continues from checkpoints.
#===============================================================================

#SBATCH --job-name=nig_mgpu
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/nig_mgpu_%A_%a.out
#SBATCH --error=logs/slurm/nig_mgpu_%A_%a.err

#===============================================================================
# Environment Setup
#===============================================================================
PROJECT_DIR="${GRAPHPERF_RT_ROOT}"
NIG_DIR="${PROJECT_DIR}/nig_experiment"
PYTHON="/projects/bdau/envs/cfg_env/bin/python"
CONFIG_FILE="${NIG_DIR}/hyperparam_search/configs_latest.json"

cd "${NIG_DIR}"
mkdir -p logs/slurm hyperparam_search/results

ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
CONFIG_OFFSET=$((ARRAY_ID * 8))

#===============================================================================
# Print Info
#===============================================================================
echo "=============================================="
echo "NIG Multi-GPU Hyperparameter Search"
echo "=============================================="
echo "Array Task:  ${ARRAY_ID}"
echo "Configs:     ${CONFIG_OFFSET} to $((CONFIG_OFFSET + 7))"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURM_NODELIST}"
echo "GPUs:        8"
echo "Start:       $(date)"
echo "=============================================="

# Get total configs
TOTAL_CONFIGS=$($PYTHON -c "
import json
with open('${CONFIG_FILE}') as f:
    print(json.load(f)['total_configs'])
")

echo "Total configs in search: ${TOTAL_CONFIGS}"
echo ""

#===============================================================================
# Launch 8 Training Jobs in Parallel
#===============================================================================
PIDS=()

for GPU_ID in {0..7}; do
    TASK_ID=$((CONFIG_OFFSET + GPU_ID))

    # Skip if beyond total configs
    if [ $TASK_ID -ge $TOTAL_CONFIGS ]; then
        echo "GPU ${GPU_ID}: Skipping (task ${TASK_ID} >= ${TOTAL_CONFIGS})"
        continue
    fi

    # Get config
    CONFIG=$($PYTHON -c "
import json
with open('${CONFIG_FILE}') as f:
    c = json.load(f)['configs'][${TASK_ID}]
print(f\"{c['experiment_name']}|{c['lambda_mse']}|{c['lambda_ns']}|{c['lr']}|{c['hidden_dim']}|{c['epochs']}|{c['batch_size']}\")
")

    IFS='|' read -r EXP_NAME LAMBDA_MSE LAMBDA_NS LR HIDDEN_DIM EPOCHS BATCH_SIZE <<< "$CONFIG"

    echo "GPU ${GPU_ID}: Task ${TASK_ID} - ${EXP_NAME}"

    # Launch in background
    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID

        $PYTHON train.py \
            --experiment "${EXP_NAME}" \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --hidden_dim ${HIDDEN_DIM} \
            --lambda_mse ${LAMBDA_MSE} \
            --lambda_ns ${LAMBDA_NS} \
            --device cuda \
            > logs/slurm/task_${TASK_ID}_gpu${GPU_ID}.log 2>&1

        EC=$?

        # Save summary
        RESULTS="${NIG_DIR}/experiments/${EXP_NAME}/results.json"
        SUMMARY="${NIG_DIR}/hyperparam_search/results/task_${TASK_ID}.json"

        if [ -f "${RESULTS}" ]; then
            $PYTHON -c "
import json
with open('${RESULTS}') as f:
    r = json.load(f)
r['task_id'] = ${TASK_ID}
r['gpu_id'] = ${GPU_ID}
r['experiment_name'] = '${EXP_NAME}'
with open('${SUMMARY}', 'w') as f:
    json.dump(r, f, indent=2)
print(f'Task ${TASK_ID} (GPU ${GPU_ID}): R²={r.get(\"test_r2\", 0):.4f}')
"
        else
            echo "{\"task_id\": ${TASK_ID}, \"status\": \"failed\", \"exit_code\": ${EC}}" > "${SUMMARY}"
            echo "Task ${TASK_ID} (GPU ${GPU_ID}): FAILED"
        fi
    ) &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting..."
echo "=============================================="

# Wait for all
for PID in "${PIDS[@]}"; do
    wait $PID
done

#===============================================================================
# Summary
#===============================================================================
echo ""
echo "=============================================="
echo "All jobs completed at $(date)"
echo "=============================================="
echo ""

# Quick summary
for GPU_ID in {0..7}; do
    TASK_ID=$((CONFIG_OFFSET + GPU_ID))
    SUMMARY="${NIG_DIR}/hyperparam_search/results/task_${TASK_ID}.json"
    if [ -f "${SUMMARY}" ]; then
        $PYTHON -c "
import json
with open('${SUMMARY}') as f:
    r = json.load(f)
status = 'OK' if r.get('test_r2') else 'FAIL'
r2 = r.get('test_r2', 0)
print(f'  Task {r[\"task_id\"]:2d}: {status} R²={r2:.4f}')
" 2>/dev/null
    fi
done

echo ""
echo "Results: ${NIG_DIR}/hyperparam_search/results/"
echo "=============================================="
