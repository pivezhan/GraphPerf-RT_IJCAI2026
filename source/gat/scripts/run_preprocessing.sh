#!/bin/bash
#SBATCH --job-name=preprocess_v2
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/logs/preprocess_%j.out
#SBATCH --error=experiments/logs/preprocess_%j.err

set -e  # Exit on error

CONFIG="configs/preprocess_config_v2.yaml"

echo "========================================"
echo "Multi-Platform Preprocessing Pipeline V2"
echo "========================================"
echo ""

# Check if raw data exists
echo "Checking raw data files..."
if [ ! -f "data/raw/v2/profiling_data_rubikpi_v2.csv" ]; then
    echo "❌ Error: data/raw/v2/profiling_data_rubikpi_v2.csv not found"
    exit 1
fi

if [ ! -f "data/raw/v2/profiling_data_tx2_v2.csv" ]; then
    echo "❌ Error: data/raw/v2/profiling_data_tx2_v2.csv not found"
    exit 1
fi

echo "✓ Raw data files found"
echo ""

# Step 1: Process raw data (parse, encode, pad, merge)
echo "========================================"
echo "Step 1: Processing Raw Data"
echo "========================================"
python scripts/01_process_data_v2.py --config $CONFIG
echo ""

# Step 2: Normalize merged data (global normalization)
echo "========================================"
echo "Step 2: Normalizing Data"
echo "========================================"
python scripts/02_normalize_data_v2.py --config $CONFIG
echo ""

# Step 3: Split into train/val/test (NORMAL - without log transform)
echo "========================================"
echo "Step 3: Creating Splits (Normal)"
echo "========================================"
python scripts/03_split_data.py \
    --input data/processed/v2/merged_normalized.csv \
    --ratios 0.7 0.15 0.15 \
    --output-dir data/splits/v2 \
    --seed 42
echo ""

# Step 4: Create log-transformed splits (RECOMMENDED for GAT)
echo "========================================"
echo "Step 4: Creating Log-Transformed Splits"
echo "========================================"
python scripts/create_log_splits_v2.py \
    --input data/processed/v2/merged_processed.csv \
    --output-dir data/splits/v2 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
echo ""

echo "========================================"
echo "Preprocessing Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  Processed:"
echo "    - data/processed/v2/rubikpi_processed.csv"
echo "    - data/processed/v2/tx2_processed.csv"
echo "    - data/processed/v2/merged_processed.csv"
echo "    - data/processed/v2/merged_normalized.csv"
echo ""
echo "  Normal splits:"
echo "    - data/splits/v2/train.csv"
echo "    - data/splits/v2/val.csv"
echo "    - data/splits/v2/test.csv"
echo ""
echo "  Log-transformed splits (RECOMMENDED for GAT):"
echo "    - data/splits/v2/train_log.csv "
echo "    - data/splits/v2/val_log.csv "
echo "    - data/splits/v2/test_log.csv "
echo ""
echo "Ready for training!"
echo "Use LOG-TRANSFORMED splits for best GAT performance!"