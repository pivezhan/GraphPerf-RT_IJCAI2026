"""
PyTorch Geometric Dataset for GAT-based performance prediction (V2).
Uses ALL available features including input_arg and performance counters.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import Tuple, List, Optional
import json


class CoreGraphDatasetV2(Dataset):
    """Dataset V2 with ALL features: input_arg, performance counters, etc."""
    
    def __init__(self, 
                 csv_path: str,
                 metadata_path: Optional[str] = None,
                 num_cores: int = 8,
                 target_column: str = 'time_elapsed',
                 use_cfg_features: bool = True):
        """Initialize dataset with V2 features."""
        self.num_cores = num_cores
        self.df = pd.read_csv(csv_path)
        self.use_cfg_features = use_cfg_features
        
        # Load metadata if provided
        self.metadata = None
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Define feature columns
        self.node_feature_cols = self._get_node_feature_columns()
        self.global_feature_cols = self._get_global_feature_columns()
        self.target_col = target_column
        
        # Create edge index (fully connected graph)
        self.edge_index = self._create_edge_index()
        
        print(f"Dataset V2 initialized:")
        print(f"  Samples: {len(self.df)}")
        print(f"  Node features per core: {len(self.node_feature_cols) // num_cores}")
        print(f"  Global features: {len(self.global_feature_cols)}")
        print(f"  Target: {self.target_col}")
        print(f"  Graph: {num_cores} nodes, fully connected")
        print(f"  CFG features: {'enabled' if use_cfg_features else 'disabled'}")
    
    def _get_node_feature_columns(self) -> List[str]:
        """Get node feature column names (per-core features)."""
        node_features = []
        
        # Core activation status (8 cores)
        for i in range(self.num_cores):
            node_features.append(f'core_active_{i}')
        
        # Frequency level (8 cores)
        for i in range(self.num_cores):
            node_features.append(f'freq_lvl_{i}')
        
        # Frequency in Hz (8 cores)
        for i in range(self.num_cores):
            node_features.append(f'freq_hz_{i}')
        
        return node_features
    
    def _get_global_feature_columns(self) -> List[str]:
        """Get ALL global feature column names (V2 - with one-hot encoding)."""
        global_features = []
        
        # ===== 1. BASE NUMERICAL FEATURES =====
        base_features = [
            'frequency_index',
            'num_cores',
            'cores_str',
            'avg_temp_before',
            'thermal_init_mean'
        ]
        for feat in base_features:
            if feat in self.df.columns:
                global_features.append(feat)
        
        # ===== 2. CRITICAL: PROBLEM SIZE =====
        if 'input_arg' in self.df.columns:
            global_features.append('input_arg')
        
        # ===== 3. PERFORMANCE COUNTERS =====
        perf_counters = [
            'branch_misses', 'cache_misses', 'cycles', 'instructions',
            'cache_references', 'branch_instructions', 'context_switches',
            'minor_faults', 'major_faults', 'page_faults',
            'task_clock', 'cpu_clock', 'user', 'sys'
        ]
        for counter in perf_counters:
            if counter in self.df.columns:
                global_features.append(counter)
        
        # ===== 4. ONE-HOT ENCODED CATEGORICALS =====
        # Benchmark (bench_0 to bench_41)
        bench_cols = [col for col in self.df.columns if col.startswith('bench_')]
        global_features.extend(sorted(bench_cols))
        
        # Platform (plat_0, plat_1)
        plat_cols = [col for col in self.df.columns if col.startswith('plat_')]
        global_features.extend(sorted(plat_cols))
        
        # Run mode (mode_0, mode_1)
        mode_cols = [col for col in self.df.columns if col.startswith('mode_')]
        global_features.extend(sorted(mode_cols))
        
        # Variant (var_0, var_1, var_2)
        var_cols = [col for col in self.df.columns if col.startswith('var_')]
        global_features.extend(sorted(var_cols))
        
        # ===== 5. THERMAL FEATURES =====
        thermal_cols = [col for col in self.df.columns 
                    if col.startswith('thermal_zone') and col.endswith('_before')]
        global_features.extend(sorted(thermal_cols))
        
        # ===== 6. OPTIONAL: CFG FEATURES =====
        if self.use_cfg_features:
            cfg_cols = [col for col in self.df.columns if col.startswith('cfg_')]
            if cfg_cols:
                print(f"  Found {len(cfg_cols)} CFG features")
                global_features.extend(sorted(cfg_cols))
        
        # Verify all exist
        existing = [f for f in global_features if f in self.df.columns]
        missing = set(global_features) - set(existing)
        
        if missing:
            print(f"  Warning: {len(missing)} features not found")
        
        return existing

    def _create_edge_index(self) -> torch.Tensor:
        """Create fully connected edge index."""
        edges = []
        for i in range(self.num_cores):
            for j in range(self.num_cores):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _create_node_features(self, row: pd.Series) -> torch.Tensor:
        """Create node feature matrix [num_nodes, 3]."""
        node_features = []
        for i in range(self.num_cores):
            core_active = row[f'core_active_{i}']
            freq_lvl = row[f'freq_lvl_{i}']
            freq_hz = row[f'freq_hz_{i}']
            node_features.append([core_active, freq_lvl, freq_hz])
        return torch.tensor(node_features, dtype=torch.float32)
    
    def _create_global_features(self, row: pd.Series) -> torch.Tensor:
        """Create global feature vector with ALL V2 features."""
        global_vals = []
        for col in self.global_feature_cols:
            val = row[col]
            # Handle any NaN values
            if pd.isna(val):
                val = 0.0
            global_vals.append(val)
        return torch.tensor(global_vals, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Data:
        """Get a single graph sample."""
        row = self.df.iloc[idx]
        
        # Create graph components
        x = self._create_node_features(row)
        global_feat = self._create_global_features(row)
        y = torch.tensor([row[self.target_col]], dtype=torch.float32)
        
        # Store global features with special handling
        data = Data(
            x=x,
            edge_index=self.edge_index,
            y=y
        )
        
        # Add global features as a 2D tensor [1, feature_dim]
        data.global_features = global_feat.unsqueeze(0)
        
        return data
    
    def get_feature_dims(self) -> Tuple[int, int]:
        """Get feature dimensions."""
        node_feature_dim = 3
        global_feature_dim = len(self.global_feature_cols)
        return node_feature_dim, global_feature_dim


# Custom collate function to handle global features correctly
def collate_fn(data_list):
    """Custom collate to properly batch global features."""
    # Use PyG's default batching for graph components
    batch = Batch.from_data_list(data_list)
    
    # Manually stack global features [batch_size, global_dim]
    global_features = torch.cat([d.global_features for d in data_list], dim=0)
    batch.global_features = global_features
    
    return batch


def create_dataloaders_v2(train_path: str,
                          val_path: str,
                          test_path: str,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          metadata_path: Optional[str] = None,
                          target_column: str = 'time_elapsed',
                          use_cfg_features: bool = True) -> Tuple:
    """Create DataLoaders with V2 dataset (all features)."""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CoreGraphDatasetV2(
        train_path, metadata_path, 
        target_column=target_column,
        use_cfg_features=use_cfg_features
    )
    val_dataset = CoreGraphDatasetV2(
        val_path, metadata_path, 
        target_column=target_column,
        use_cfg_features=use_cfg_features
    )
    test_dataset = CoreGraphDatasetV2(
        test_path, metadata_path, 
        target_column=target_column,
        use_cfg_features=use_cfg_features
    )
    
    # Create dataloaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Get feature dimensions
    feature_dims = train_dataset.get_feature_dims()
    
    print("\nDataLoaders V2 created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Feature dims: node={feature_dims[0]}, global={feature_dims[1]}")
    
    return train_loader, val_loader, test_loader, feature_dims


if __name__ == '__main__':
    print("Testing CoreGraphDatasetV2...")
    
    # Test with V2 data
    dataset = CoreGraphDatasetV2(
        csv_path='data/splits/v2/train_log.csv',
        metadata_path='data/processed/v2/merged_metadata.json'
    )
    
    sample = dataset[0]
    
    print("\nSample graph structure:")
    print(f"  Node features (x): {sample.x.shape}")
    print(f"  Edge index: {sample.edge_index.shape}")
    print(f"  Global features: {sample.global_features.shape}")
    print(f"  Target (y): {sample.y.shape}")
    
    print("\nNode features for first 3 cores:")
    print(sample.x[:3])
    
    print("\nGlobal feature dimensions:")
    print(f"  Total: {sample.global_features.shape[1]} features")
    
    print("\nTarget:")
    print(sample.y)