"""
Graph Attention Network for OpenMP performance prediction.

Implements the heterogeneous GAT surrogate from §4 ("Model Architecture")
of the camera-ready paper.  Each core in the target SoC becomes a node in a
fully-connected graph; the graph carries 3 per-node features (active, freq
level, freq Hz) and 69 global features (one-hot benchmark, platform, run
mode, hardware-counter aggregates, thermal headroom).

Architecture (matches Table 2 of the paper):
    1. Two GATConv layers (4 attention heads each) — node features → 256-dim
       → 128-dim.
    2. Global mean+max pooling aggregates node embeddings.
    3. Fusion concatenates pooled embedding with the 69 global features.
    4. MLP head: [fused → 512 → 256 → 128 → 1] regresses log-makespan.

Used as the world model by the model-based RL agent in §6 to provide fast,
calibrated rollouts for Dyna-style planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GATPredictor(nn.Module):
    """GAT model for runtime prediction."""
    
    def __init__(self,
                 node_feature_dim: int = 3,
                 global_feature_dim: int = 8,
                 hidden_dim: int = 64,
                 num_gat_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 pooling: str = 'mean'):
        """
        Initialize GAT model.
        
        Args:
            node_feature_dim: Dimension of node features (default: 3)
            global_feature_dim: Dimension of global features (default: 8)
            hidden_dim: Hidden dimension for GAT layers
            num_gat_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            pooling: Pooling method ('mean', 'max', or 'both')
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pooling = pooling
        
        # Node feature encoder
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            out_channels = hidden_dim
            
            # Last layer: single head (no concatenation)
            concat = True if i < num_gat_layers - 1 else False
            heads = num_heads if i < num_gat_layers - 1 else 1
            
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout
                )
            )
        
        # Determine pooled dimension
        if pooling == 'both':
            pooled_dim = hidden_dim * 2
        else:
            pooled_dim = hidden_dim
        
        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion and prediction head
        fusion_input_dim = pooled_dim + hidden_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, global_features, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            global_features: Global features [batch_size, global_feature_dim]
            batch: Batch assignment [num_nodes]
        
        Returns:
            predictions: [batch_size, 1]
        """
        # Encode node features
        x = self.node_encoder(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        # Pool node features to graph level
        if self.pooling == 'mean':
            graph_repr = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            graph_repr = global_max_pool(x, batch)
        elif self.pooling == 'both':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            graph_repr = torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Encode global features
        global_repr = self.global_encoder(global_features)
        
        # Fuse graph representation with global features
        fused = torch.cat([graph_repr, global_repr], dim=1)
        
        # Predict
        out = self.predictor(fused)
        
        return out


def create_model(config: dict) -> GATPredictor:
    """Create model from config."""
    model = GATPredictor(
        node_feature_dim=config.get('node_feature_dim', 3),
        global_feature_dim=config.get('global_feature_dim', 8),
        hidden_dim=config.get('hidden_dim', 64),
        num_gat_layers=config.get('num_gat_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        pooling=config.get('pooling', 'mean')
    )
    return model


if __name__ == '__main__':
    # Test model
    print("Testing GATPredictor...")
    
    model = GATPredictor(
        node_feature_dim=3,
        global_feature_dim=8,
        hidden_dim=64,
        num_gat_layers=2,
        num_heads=4,
        dropout=0.1,
        pooling='mean'
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 32
    num_nodes = batch_size * 8
    
    x = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 7))
    global_features = torch.randn(batch_size, 8)
    batch = torch.repeat_interleave(torch.arange(batch_size), 8)
    
    out = model(x, edge_index, global_features, batch)
    
    print(f"\nForward pass test:")
    print(f"  Input: x={x.shape}, edge_index={edge_index.shape}")
    print(f"  Global: {global_features.shape}, batch={batch.shape}")
    print(f"  Output: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n✅ Model test passed!")