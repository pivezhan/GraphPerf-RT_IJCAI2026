#!/usr/bin/env python3
"""
Step 1: Data Processing V2 - Full Feature Set with Platform Adaptation

Handles multi-platform with different architectures:
- Different core counts (6 vs 8)
- Different thermal zone counts (8 vs 36)
- Pads arrays to maximum length across platforms
- Includes ALL performance counters

Usage:
    python scripts/01_process_data.py --config configs/preprocess_config_v2.yaml
"""

import argparse
import yaml
import json
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class MultiPlatformProcessor:
    """Process raw data with platform adaptation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_encoders = {}
        self.max_cores = config['platform_adaptation']['max_cores']
        self.max_thermal_zones = config['platform_adaptation']['max_thermal_zones']
        self.padding_value = config['platform_adaptation']['padding_value']
    
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """Load raw CSV data."""
        print(f"Loading: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  → {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def parse_array_column(self, df: pd.DataFrame, column: str, 
                          config: Dict[str, Any],
                          platform_name: str) -> pd.DataFrame:
        """Parse array string columns with platform-specific lengths and padding."""
        prefix = config['output_prefix']
        # Handle both 'length' and 'max_length' for compatibility
        max_length = config.get('max_length', config.get('length', 8))
        dtype = config['dtype']
        
        # Get platform-specific length
        platform_lengths = config.get('platform_lengths', {})
        expected_length = platform_lengths.get(platform_name, max_length)
        
        def safe_parse_and_pad(x):
            try:
                if isinstance(x, str):
                    arr = ast.literal_eval(x)
                else:
                    arr = x if isinstance(x, list) else [0] * expected_length
                
                # Pad to max_length
                if len(arr) < max_length:
                    arr = arr + [self.padding_value] * (max_length - len(arr))
                elif len(arr) > max_length:
                    arr = arr[:max_length]
                
                return arr
            except:
                return [self.padding_value] * max_length
        
        parsed = df[column].apply(safe_parse_and_pad)
        
        for i in range(max_length):
            df[f"{prefix}_{i}"] = parsed.apply(
                lambda x: x[i]
            ).astype(dtype)
        
        print(f"  → Parsed {column}: {expected_length} → {max_length} (padded)")
        
        return df
    
    def parse_core_combination(self, df: pd.DataFrame, 
                               config: Dict[str, Any]) -> pd.DataFrame:
        """Parse core_combination into binary mask with padding."""
        prefix = config['output_prefix']
        # Handle both 'length' and 'max_length' for compatibility
        max_length = config.get('max_length', config.get('length', 8))
        dtype = config['dtype']
        
        def to_binary_mask(core_str):
            mask = [0] * max_length
            if pd.isna(core_str):
                return mask
            
            # Handle both formats: "1234567" or "1,2,3,4,5,6,7"
            core_str = str(core_str).replace(',', '')
            cores = [int(c) for c in core_str if c.isdigit()]
            
            for core_id in cores:
                if 1 <= core_id <= max_length:
                    mask[core_id - 1] = 1
            
            return mask
        
        masks = df['core_combination'].apply(to_binary_mask)
        for i in range(max_length):
            df[f"{prefix}_{i}"] = masks.apply(lambda x: x[i]).astype(dtype)
        
        return df
    
    def parse_cores_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse cores_str to count number of cores."""
        def count_cores(core_str):
            if pd.isna(core_str):
                return 0
            core_str = str(core_str)
            if ',' in core_str:
                return len(core_str.split(','))
            elif core_str.strip():
                return len([c for c in core_str if c.isdigit()])
            else:
                return 0
        
        df['cores_str'] = df['cores_str'].apply(count_cores).astype('int8')
        return df
    
    def aggregate_thermal_zones(self, df: pd.DataFrame, 
                                config: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate thermal zones into single mean value."""
        prefix = config['input_prefix']
        output_col = config['output_column']
        dtype = config['dtype']
        
        thermal_cols = [col for col in df.columns if col.startswith(prefix)]
        
        if not thermal_cols:
            print(f"  ⚠️  No thermal columns found")
            df[output_col] = 0.0
            return df
        
        print(f"  → Found {len(thermal_cols)} thermal zones")
        
        def thermal_mean(row):
            values = row[thermal_cols].values
            non_zero = values[values > 0]
            return non_zero.mean() if len(non_zero) > 0 else 0.0
        
        df[output_col] = df.apply(thermal_mean, axis=1).astype(dtype)
        return df
    
    def encode_categorical(self, df: pd.DataFrame, 
                          config: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical variables."""
        for feature, feat_config in config.items():
            if feat_config.get('optional', False) and feature not in df.columns:
                print(f"  ⚠️  Optional '{feature}' not found, skipping")
                continue
            
            if feature not in df.columns:
                continue
            
            df[feature] = df[feature].astype(str).str.strip()
            dtype = feat_config['dtype']
            
            if 'mapping' in feat_config:
                mapping = feat_config['mapping']
                df[feature] = df[feature].map(mapping)
                unmapped = df[feature].isna().sum()
                if unmapped > 0:
                    print(f"  ⚠️  {unmapped} unmapped '{feature}'")
                df[feature] = df[feature].fillna(-1).astype(dtype)
                self.label_encoders[feature] = mapping
            
            elif feat_config.get('auto_detect', False):
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                df[feature] = df[feature].astype(dtype)
                self.label_encoders[feature] = dict(
                    zip(le.classes_, le.transform(le.classes_))
                )
                print(f"  → Encoded '{feature}': {len(le.classes_)} categories")
        
        return df
    
    def process_numerical_features(self, df: pd.DataFrame, 
                                   config: Dict[str, Any],
                                   feature_type: str) -> pd.DataFrame:
        """Process numerical features with optional and missing value handling."""
        for feature, feat_config in config.items():
            dtype = feat_config['dtype']
            optional = feat_config.get('optional', False)
            fill_missing = feat_config.get('fill_missing', 0)
            
            if feature not in df.columns:
                if optional:
                    print(f"  ⚠️  Optional '{feature}' not found, filling with {fill_missing}")
                    df[feature] = fill_missing
                else:
                    print(f"  ⚠️  Required '{feature}' not found!")
                    continue
            
            # Special handling for input_arg (mixed numeric/string)
            if feature == 'input_arg':
                df[feature] = self.convert_input_arg_to_numeric(df[feature])
                print(f"  → Converted 'input_arg' to numeric (range: {df[feature].min():.0f} to {df[feature].max():.0f})")
            
            # Handle missing values
            df[feature] = df[feature].fillna(fill_missing)
            
            # Cast type
            try:
                df[feature] = df[feature].astype(dtype)
            except Exception as e:
                print(f"  ⚠️  Failed to cast '{feature}' to {dtype}: {e}")
                print(f"      Attempting conversion with coercion...")
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(fill_missing).astype('float32')
        
        return df
    
    def convert_input_arg_to_numeric(self, series: pd.Series) -> pd.Series:
        """
        Convert input_arg to numeric values.
        
        Handles:
        - Numeric strings: "100000" → 100000
        - MINI_DATASET → 1000 (small problem)
        - LARGE_DATASET → 1000000 (large problem)
        - Protein files (prot.XX.aa) → parse XX as size
        - Other strings → hash to reasonable range
        """
        def parse_value(val):
            if pd.isna(val):
                return 0
            
            # Try direct numeric conversion
            try:
                return float(val)
            except:
                pass
            
            val_str = str(val).strip()
            
            # Handle known dataset sizes
            if val_str.upper() == 'MINI_DATASET':
                return 1000
            elif val_str.upper() == 'SMALL_DATASET':
                return 10000
            elif val_str.upper() == 'MEDIUM_DATASET':
                return 100000
            elif val_str.upper() == 'LARGE_DATASET':
                return 1000000
            
            # Handle protein files: "prot.20.aa" → 20
            if 'prot' in val_str.lower() and '.aa' in val_str.lower():
                try:
                    # Extract number from "prot.XX.aa"
                    parts = val_str.split('.')
                    for part in parts:
                        if part.isdigit():
                            return float(part) * 1000  # Scale to reasonable range
                except:
                    pass
            
            # Last resort: hash to consistent value
            # Use hash to get consistent mapping
            hash_val = abs(hash(val_str)) % 1000000
            return float(hash_val)
        
        return series.apply(parse_value)
    
    def clean_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Clean data."""
        initial = len(df)
        
        if config.get('remove_duplicates', True):
            df = df.drop_duplicates()
        
        if config.get('drop_na', True):
            # Only drop rows where target is missing
            if 'time_elapsed' in df.columns:
                df = df.dropna(subset=['time_elapsed'])
        
        if 'constraints' in config:
            for col, limits in config['constraints'].items():
                if col in df.columns:
                    if 'min' in limits:
                        df = df[df[col] >= limits['min']]
                    if 'max' in limits:
                        df = df[df[col] <= limits['max']]
        
        removed = initial - len(df)
        if removed > 0:
            print(f"  → Removed {removed} rows ({removed/initial*100:.1f}%)")
        
        return df
    
    def get_final_columns(self) -> List[str]:
        """Get final column order."""
        cols = []
        
        # Categorical
        for feature in self.config['features']['categorical'].keys():
            cols.append(feature)
        
        # Numerical before
        for feature in self.config['features']['numerical_before'].keys():
            cols.append(feature)
        
        # Performance counters
        for feature in self.config['features']['performance_counters'].keys():
            cols.append(feature)
        
        # Thermal
        cols.append(self.config['features']['thermal']['output_column'])
        
        # Arrays
        for feat_config in self.config['features']['arrays'].values():
            prefix = feat_config['output_prefix']
            max_length = feat_config.get('max_length', feat_config.get('length', 8))
            cols.extend([f"{prefix}_{i}" for i in range(max_length)])
        
        # Special (core combination)
        special_config = self.config['features']['special']['core_combination']
        prefix = special_config['output_prefix']
        max_length = special_config.get('max_length', special_config.get('length', 8))
        cols.extend([f"{prefix}_{i}" for i in range(max_length)])
        
        # Target
        for target in self.config['targets']:
            cols.append(target['name'])
        
        return cols
    
    def process_dataset(self, dataset_name: str, dataset_config: Dict) -> pd.DataFrame:
        """Process a single dataset with platform-specific handling."""
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        input_path = dataset_config['raw_data']
        output_path = dataset_config['processed_data']
        platform_name = dataset_config['platform_name']
        
        # Load
        df = self.load_raw_data(input_path)
        
        # Parse arrays (with platform-specific lengths)
        for feature, feat_config in self.config['features']['arrays'].items():
            if feature in df.columns:
                df = self.parse_array_column(df, feature, feat_config, platform_name)
        
        # Parse special features
        if 'core_combination' in df.columns:
            df = self.parse_core_combination(
                df, self.config['features']['special']['core_combination']
            )
        
        if 'cores_str' in df.columns:
            df = self.parse_cores_str(df)
        
        # Aggregate thermal
        df = self.aggregate_thermal_zones(df, self.config['features']['thermal'])
        
        # Encode categorical
        df = self.encode_categorical(df, self.config['features']['categorical'])
        
        # Process numerical features (before execution)
        df = self.process_numerical_features(
            df, self.config['features']['numerical_before'], 'before'
        )
        
        # Process performance counters (after execution)
        df = self.process_numerical_features(
            df, self.config['features']['performance_counters'], 'performance'
        )
        
        # Clean
        df = self.clean_data(df, self.config['cleaning'])
        
        # Select final columns
        final_cols = self.get_final_columns()
        available_cols = [col for col in final_cols if col in df.columns]
        
        if len(available_cols) < len(final_cols):
            missing = set(final_cols) - set(available_cols)
            print(f"  ⚠️  Missing {len(missing)} columns: {list(missing)[:5]}...")
        
        df_final = df[available_cols].copy()
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False, float_format='%.6f')
        
        print(f"  ✅ Saved: {output_path}")
        print(f"     Rows: {len(df_final):,} | Columns: {len(df_final.columns)}")
        
        return df_final


def to_json_serializable(obj):
    """Convert numpy types to JSON serializable."""
    if isinstance(obj, dict):
        return {key: to_json_serializable(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description="Process multi-platform data with adaptation"
    )
    parser.add_argument('--config', type=str, 
                       default='configs/preprocess_config_v2.yaml',
                       help='Config file path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = MultiPlatformProcessor(config)
    
    print(f"\n{'#'*60}")
    print("# Multi-Platform Data Processing V2")
    print(f"# Max cores: {processor.max_cores}")
    print(f"# Max thermal zones: {processor.max_thermal_zones}")
    print(f"{'#'*60}\n")
    
    # Process each dataset
    processed_dfs = []
    
    for dataset_name, dataset_config in config['datasets'].items():
        raw_path = Path(dataset_config['raw_data'])
        
        if not raw_path.exists():
            print(f"⚠️  Skipping {dataset_name}: {raw_path} not found")
            continue
        
        df = processor.process_dataset(dataset_name, dataset_config)
        processed_dfs.append(df)
    
    # Merge datasets
    if len(processed_dfs) > 1:
        print(f"\n{'='*60}")
        print("Merging processed datasets")
        print(f"{'='*60}")
        
        # Check column alignment
        col_sets = [set(df.columns) for df in processed_dfs]
        common_cols = col_sets[0].intersection(*col_sets[1:])
        
        if len(common_cols) < len(processed_dfs[0].columns):
            print(f"  ⚠️  Column mismatch detected!")
            for i, df in enumerate(processed_dfs):
                print(f"     Dataset {i}: {len(df.columns)} cols")
        
        # Merge with common columns
        df_merged = pd.concat([df[sorted(common_cols)] for df in processed_dfs], 
                              ignore_index=True)
        
        merged_path = config['merged']['processed_data']
        Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(merged_path, index=False, float_format='%.6f')
        
        print(f"  ✅ Merged: {merged_path}")
        print(f"     Total rows: {len(df_merged):,}")
        print(f"     Columns: {len(df_merged.columns)}")
        
        # Platform distribution
        print(f"\n  Platform distribution:")
        for platform, count in df_merged['platform'].value_counts().sort_index().items():
            platform_name = {0: 'RubikPi', 1: 'TX2', 2: 'NX'}.get(platform, platform)
            print(f"    {platform_name}: {count:,} ({count/len(df_merged)*100:.1f}%)")
    
    # Save metadata
    metadata = {
        'label_encoders': to_json_serializable(processor.label_encoders),
        'final_columns': processor.get_final_columns(),
        'num_datasets': len(processed_dfs),
        'total_samples': sum(len(df) for df in processed_dfs),
        'platform_adaptation': {
            'max_cores': processor.max_cores,
            'max_thermal_zones': processor.max_thermal_zones,
            'padding_value': processor.padding_value
        },
        'gat_structure': config['gat_structure']
    }
    
    metadata_path = config['merged']['metadata'].replace('.json', '_processed.json')
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Processing Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {metadata['total_samples']:,}")
    print(f"Features: {len(metadata['final_columns'])}")
    print(f"Metadata: {metadata_path}\n")


if __name__ == '__main__':
    main()