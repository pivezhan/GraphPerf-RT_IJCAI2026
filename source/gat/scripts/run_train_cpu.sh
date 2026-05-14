#!/bin/bash
#SBATCH --job-name=gat_v1_cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/logs/train_v1_%j.out
#SBATCH --error=experiments/logs/train_v1_%j.err

# Change to project directory
cd ${GRAPHPERF_RT_ROOT}/gat || exit 1

# ============================================================================
# Direct Python Path (MOST RELIABLE METHOD)
# ============================================================================
PYTHON="$HOME/.conda/envs/gnn4_env/bin/python"

# Verify Python exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Please check your conda environment path"
    exit 1
fi

echo "========================================="
echo "Environment Information"
echo "========================================="
echo "Python: $PYTHON"
$PYTHON --version
echo ""

# Set CPU optimization flags
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Print job info
echo "========================================="
echo "Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Start time: $(date)"
echo ""

# Verify required files exist
echo "========================================="
echo "Checking Required Files"
echo "========================================="
ERRORS=0

if [ ! -f "data/splits/v1/train_log.csv" ]; then
    echo "✗ data/splits/v1/train_log.csv NOT FOUND"
    ERRORS=$((ERRORS+1))
else
    echo "✓ data/splits/v1/train_log.csv"
fi

if [ ! -f "data/splits/v1/val_log.csv" ]; then
    echo "✗ data/splits/v1/val_log.csv NOT FOUND"
    ERRORS=$((ERRORS+1))
else
    echo "✓ data/splits/v1/val_log.csv"
fi

if [ ! -f "configs/model_config_v2_log.yaml" ]; then
    echo "✗ configs/model_config_v2_log.yaml NOT FOUND"
    ERRORS=$((ERRORS+1))
else
    echo "✓ configs/model_config_v1_log.yaml"
fi

if [ ! -f "scripts/train_v2.py" ]; then
    echo "✗ scripts/train_v2.py NOT FOUND"
    ERRORS=$((ERRORS+1))
else
    echo "✓ scripts/train_v2.py"
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "ERROR: $ERRORS required file(s) missing. Exiting."
    exit 1
fi
echo ""

# Print dataset info
echo "========================================="
echo "Dataset Information"
echo "========================================="
$PYTHON -c "
import pandas as pd
train = pd.read_csv('data/splits/v1/train_log.csv')
val = pd.read_csv('data/splits/v1/val_log.csv')
print(f'Train: {len(train):,} samples, {len(train.columns)} features')
print(f'Val:   {len(val):,} samples, {len(val.columns)} features')
print(f'Target column: time_elapsed_log')
print(f'Target range: [{train[\"time_elapsed_log\"].min():.3f}, {train[\"time_elapsed_log\"].max():.3f}]')
"
echo ""

# Run training
echo "========================================="
echo "Starting Training"
echo "========================================="
srun $PYTHON scripts/train_v2.py \
    --config configs/model_config_v2_log.yaml \
    --experiment exp_v1_001_gat_cpu_log

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training Completed Successfully!"
else
    echo "✗ Training Failed (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "========================================="

exit $EXIT_CODE