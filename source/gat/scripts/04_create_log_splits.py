#!/usr/bin/env python3
"""
Create train/val/test splits with log-transformed target and proper one-hot encoding.
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_stratification_key(df: pd.DataFrame) -> pd.Series:
    """Create stratification key from platform and benchmark."""
    return df['platform'].astype(str) + '_' + df['benchmark'].astype(str)


def main():
    parser = argparse.ArgumentParser(description="Create log-transformed splits with one-hot encoding")
    parser.add_argument('--input', type=str, default='data/processed/v2/merged_processed.csv')
    parser.add_argument('--output-dir', type=str, default='data/splits/v2')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Creating Log-Transformed Splits with One-Hot Encoding")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  → {len(df):,} rows, {len(df.columns)} columns")
    
    # Check target
    if 'time_elapsed' not in df.columns:
        raise ValueError("Missing 'time_elapsed' column!")
    
    print(f"\nOriginal time_elapsed: [{df['time_elapsed'].min():.4f}, {df['time_elapsed'].max():.4f}]")
    
    # Add log transform
    df['time_elapsed_log'] = np.log1p(df['time_elapsed'])
    print(f"Log-transformed: [{df['time_elapsed_log'].min():.4f}, {df['time_elapsed_log'].max():.4f}]")
    
    # One-hot encode BEFORE splitting
    print(f"\n{'='*60}")
    print("One-Hot Encoding Categorical Features")
    print(f"{'='*60}")
    
    categorical_cols = ['benchmark', 'platform', 'run_mode', 'variant']
    
    # Count unique values BEFORE encoding
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"  {col}: {unique_vals} unique values → {unique_vals} binary features")
    
    original_cols = len(df.columns)
    
    # Create stratification key BEFORE one-hot encoding
    strat_key = create_stratification_key(df)
    
    # One-hot encode
    df = pd.get_dummies(df, columns=categorical_cols, prefix=['bench', 'plat', 'mode', 'var'], dtype='int8')
    
    print(f"\n  Before encoding: {original_cols} columns")
    print(f"  After encoding:  {len(df.columns)} columns")
    
    # Split
    print(f"\n{'='*60}")
    print(f"Splitting: {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"{'='*60}")
    
    train_df, temp_df, train_strat, temp_strat = train_test_split(
        df, strat_key,
        test_size=(args.val_ratio + args.test_ratio),
        stratify=strat_key,
        random_state=args.seed
    )
    
    test_size_adjusted = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size_adjusted,
        stratify=temp_strat,
        random_state=args.seed
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Identify columns to normalize
    print(f"\n{'='*60}")
    print("Normalizing Features")
    print(f"{'='*60}")
    
    # Get all numerical columns
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude binary columns
    binary_cols = [f'core_active_{i}' for i in range(8)] + \
                  [col for col in df.columns if col.startswith(('bench_', 'plat_', 'mode_', 'var_'))]
    
    # Exclude targets
    exclude = set(binary_cols + ['time_elapsed', 'time_elapsed_log'])
    
    # Columns to normalize
    cols_to_normalize = [col for col in numerical_cols if col not in exclude]
    cols_to_normalize.append('time_elapsed_log')
    
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Binary columns (excluded): {len(binary_cols)}")
    print(f"  Columns to normalize: {len(cols_to_normalize)}")
    
    # Fill missing values
    for col in cols_to_normalize:
        train_df[col] = train_df[col].fillna(0)
        val_df[col] = val_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
    
    # Normalize
    scaler = StandardScaler()
    train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize])
    val_df[cols_to_normalize] = scaler.transform(val_df[cols_to_normalize])
    test_df[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize])
    
    print(f"\n  ✓ Normalized")
    print(f"    time_elapsed_log: mean={train_df['time_elapsed_log'].mean():.4f}, std={train_df['time_elapsed_log'].std():.4f}")
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / 'train_log.csv', index=False, float_format='%.6f')
    val_df.to_csv(output_dir / 'val_log.csv', index=False, float_format='%.6f')
    test_df.to_csv(output_dir / 'test_log.csv', index=False, float_format='%.6f')
    
    print(f"\n{'='*60}")
    print("✅ Saved Splits")
    print(f"{'='*60}")
    print(f"  Train: {output_dir / 'train_log.csv'}")
    print(f"  Val:   {output_dir / 'val_log.csv'}")
    print(f"  Test:  {output_dir / 'test_log.csv'}")
    print(f"  Total features per split: {len(train_df.columns)}")
    
    # Save metadata
    scaler_path = output_dir.parent / 'scaler_params_log.json'
    with open(scaler_path, 'w') as f:
        json.dump({
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'std': scaler.scale_.tolist(),
                'n_samples': len(train_df),
                'n_features': len(cols_to_normalize)
            },
            'normalized_columns': cols_to_normalize,
            'binary_columns': binary_cols,
            'one_hot_encoded': categorical_cols,
            'random_seed': args.seed
        }, f, indent=2)
    
    print(f"  Metadata: {scaler_path}")
    print(f"\n{'='*60}")
    print("✅ Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()