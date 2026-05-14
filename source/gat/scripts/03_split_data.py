#!/usr/bin/env python3
"""
Step 3: Train/Val/Test Split

Creates stratified splits ensuring:
- Platform balance (RubikPi vs TX2)
- Benchmark diversity
- Similar target distributions

Usage:
    python scripts/03_split_data.py --input data/processed/merged_normalized.csv --ratios 0.7 0.15 0.15
    python scripts/03_split_data.py --input data/processed/merged_normalized.csv --ratios 0.8 0.1 0.1
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def create_stratification_key(df: pd.DataFrame) -> pd.Series:
    """
    Create stratification key combining platform and benchmark.
    
    This ensures each split has similar distribution of:
    - Platform (0=RubikPi, 1=TX2)
    - Benchmark (different workloads)
    """
    # Combine platform and benchmark
    strat_key = df['platform'].astype(str) + '_' + df['benchmark'].astype(str)
    return strat_key


def split_with_diversity(df: pd.DataFrame, 
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data with stratification on platform + benchmark.
    
    Args:
        df: Input DataFrame
        train_ratio: Training set ratio (default 0.7)
        val_ratio: Validation set ratio (default 0.15)
        test_ratio: Test set ratio (default 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Create stratification key
    strat_key = create_stratification_key(df)
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=strat_key,
        random_state=random_state
    )
    
    # Second split: val vs test
    # Recalculate stratification key for remaining data
    temp_strat_key = create_stratification_key(temp_df)
    
    # Adjust ratio for second split
    test_size_adjusted = test_ratio / (val_ratio + test_ratio)
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size_adjusted,
        stratify=temp_strat_key,
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def analyze_split_quality(train_df: pd.DataFrame, 
                          val_df: pd.DataFrame, 
                          test_df: pd.DataFrame) -> None:
    """Print statistics about split quality."""
    
    print(f"\n{'='*60}")
    print("SPLIT QUALITY ANALYSIS")
    print(f"{'='*60}\n")
    
    # Size distribution
    total = len(train_df) + len(val_df) + len(test_df)
    print("📊 Size Distribution:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
    print(f"  Total: {total:,}")
    
    # Platform distribution
    print(f"\n🖥️  Platform Distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        platform_counts = split_df['platform'].value_counts().sort_index()
        platform_pcts = (platform_counts / len(split_df) * 100).round(1)
        print(f"  {split_name:5s}: ", end="")
        for platform, count in platform_counts.items():
            pct = platform_pcts[platform]
            platform_name = "RubikPi" if platform == 0 else "TX2"
            print(f"{platform_name}={count} ({pct}%)  ", end="")
        print()
    
    # Benchmark diversity
    print(f"\n📚 Benchmark Diversity:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n_benchmarks = split_df['benchmark'].nunique()
        print(f"  {split_name:5s}: {n_benchmarks} unique benchmarks")
    
    # Target distribution
    print(f"\n🎯 Target (time_elapsed) Distribution:")
    print(f"  {'':5s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        target = split_df['time_elapsed']
        print(f"  {split_name:5s}  {target.mean():8.4f}  {target.std():8.4f}  "
              f"{target.min():8.4f}  {target.max():8.4f}")
    
    # Run mode distribution
    print(f"\n🏃 Run Mode Distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        mode_counts = split_df['run_mode'].value_counts().sort_index()
        mode_pcts = (mode_counts / len(split_df) * 100).round(1)
        print(f"  {split_name:5s}: ", end="")
        for mode, count in mode_counts.items():
            pct = mode_pcts[mode]
            mode_name = "seq" if mode == 0 else "par"
            print(f"{mode_name}={count} ({pct}%)  ", end="")
        print()
    
    # Core usage distribution
    print(f"\n💻 Core Usage Distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        core_counts = split_df['num_cores'].value_counts().sort_index()
        print(f"  {split_name:5s}: ", end="")
        for cores, count in core_counts.items():
            print(f"{cores}c={count}  ", end="")
        print()


def save_splits(train_df: pd.DataFrame, 
                val_df: pd.DataFrame, 
                test_df: pd.DataFrame,
                output_dir: str = 'data/splits') -> None:
    """Save train/val/test splits to CSV files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / 'train.csv'
    val_path = output_path / 'val.csv'
    test_path = output_path / 'test.csv'
    
    train_df.to_csv(train_path, index=False, float_format='%.6f')
    val_df.to_csv(val_path, index=False, float_format='%.6f')
    test_df.to_csv(test_path, index=False, float_format='%.6f')
    
    print(f"\n{'='*60}")
    print("✅ Splits Saved:")
    print(f"{'='*60}")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Split data into train/val/test")
    parser.add_argument('--input', type=str, 
                       default='data/processed/merged_normalized.csv',
                       help='Input normalized CSV file')
    parser.add_argument('--ratios', type=float, nargs=3,
                       default=[0.7, 0.15, 0.15],
                       help='Train/Val/Test ratios (default: 0.7 0.15 0.15)')
    parser.add_argument('--output-dir', type=str,
                       default='data/splits',
                       help='Output directory for splits')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    train_ratio, val_ratio, test_ratio = args.ratios
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print(f"❌ Error: Ratios must sum to 1.0, got {sum(args.ratios)}")
        return
    
    # Load data
    print(f"\n{'='*60}")
    print("Loading normalized data...")
    print(f"{'='*60}\n")
    
    df = pd.read_csv(args.input)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Create splits
    print(f"\n{'='*60}")
    print(f"Creating stratified splits ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})")
    print(f"{'='*60}")
    
    train_df, val_df, test_df = split_with_diversity(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=args.seed
    )
    
    # Analyze quality
    analyze_split_quality(train_df, val_df, test_df)
    
    # Save splits
    save_splits(train_df, val_df, test_df, args.output_dir)
    
    print(f"{'='*60}")
    print("✅ Splitting Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
