#!/usr/bin/env python3
"""
Step 2: Normalize Data V2

Applies StandardScaler to MERGED processed data.
Normalizes all numerical features except categorical and binary.

Usage:
    python scripts/02_normalize_data.py --config configs/preprocess_config_v2.yaml
"""

import argparse
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def identify_normalization_columns(df: pd.DataFrame, config: Dict) -> List[str]:
    """
    Identify which columns to normalize based on config.
    
    Normalize: All numerical columns
    Exclude: Categorical, binary, and explicitly excluded columns
    """
    # Get exclusions from config
    exclude_categorical = set(config['normalization'].get('exclude_categorical', []))
    exclude_binary = set(config['normalization'].get('exclude_binary', []))
    
    # Get all numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove exclusions
    cols_to_normalize = [
        col for col in numerical_cols 
        if col not in exclude_categorical 
        and col not in exclude_binary
        and col != 'time_elapsed'  # Don't normalize target yet
    ]
    
    return cols_to_normalize


def main():
    parser = argparse.ArgumentParser(description="Normalize merged processed data")
    parser.add_argument('--config', type=str,
                       default='configs/preprocess_config_v2.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print("Step 2: Normalizing Merged Data")
    print(f"{'='*60}\n")
    
    # Load merged processed data
    input_path = config['merged']['processed_data']
    output_path = config['merged']['normalized_data']
    
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  → {len(df):,} rows, {len(df.columns)} columns")
    
    # Identify columns to normalize
    cols_to_normalize = identify_normalization_columns(df, config)
    
    print(f"\nNormalization plan:")
    print(f"  Total numerical columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Columns to normalize: {len(cols_to_normalize)}")
    print(f"  Excluded columns: {len(df.columns) - len(cols_to_normalize) - 1}")  # -1 for target
    
    # Check for missing values in columns to normalize
    missing_counts = df[cols_to_normalize].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n  ⚠️  Warning: {missing_counts.sum()} missing values found")
        print(f"     Filling with 0...")
        df[cols_to_normalize] = df[cols_to_normalize].fillna(0)
    
    # Fit StandardScaler
    scaler = StandardScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    print(f"\n  ✓ Normalized {len(cols_to_normalize)} features")
    print(f"    Sample means: {df[cols_to_normalize].mean().head(5).to_dict()}")
    print(f"    Sample stds:  {df[cols_to_normalize].std().head(5).to_dict()}")
    
    # Save normalized data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"\n  ✅ Saved: {output_path}")
    
    # Save scaler parameters
    scaler_metadata = {
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
            'n_samples': len(df),
            'n_features': len(cols_to_normalize)
        },
        'normalized_columns': cols_to_normalize,
        'excluded_columns': [
            col for col in df.columns if col not in cols_to_normalize
        ]
    }
    
    metadata_path = config['merged']['metadata'].replace('.json', '_scaler.json')
    with open(metadata_path, 'w') as f:
        json.dump(scaler_metadata, f, indent=2)
    
    print(f"  ✅ Scaler metadata: {metadata_path}")
    
    print(f"\n{'='*60}")
    print("✅ Normalization Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()