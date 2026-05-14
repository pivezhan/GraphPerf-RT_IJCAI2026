"""
Simple MLP baseline for performance prediction.
Uses flattened features without graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPredictor(nn.Module):
    """Simple MLP baseline - no graph structure."""
    
    def __init__(self,
                 node_feature_dim: int = 3,
                 global_feature_dim: int = 8,
                 num_nodes: int = 8,
                 hidden_dims: list = [256, 128, 64],
                 dropout: float = 0.2):
        """
        Initialize MLP.
        
        Args:
            node_feature_dim: Features per node (3: active, freq_lvl, freq_hz)
            global_feature_dim: Global features (8)
            num_nodes: Number of nodes (8 cores)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        # Input: flattened node features + global features
        input_dim = node_feature_dim * num_nodes + global_feature_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        print(f"MLP Baseline:")
        print(f"  Input dim: {input_dim} (nodes: {node_feature_dim * num_nodes}, global: {global_feature_dim})")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Output: 1")
    
    def forward(self, x, edge_index, global_features, batch):
        """
        Forward pass - ignore edge_index and batch (for compatibility).
        
        Args:
            x: Node features [num_nodes_in_batch, node_feature_dim]
            edge_index: Ignored (for API compatibility with GAT)
            global_features: Global features [batch_size, global_feature_dim]
            batch: Node-to-graph assignment [num_nodes_in_batch]
        
        Returns:
            predictions: [batch_size, 1]
        """
        batch_size = global_features.shape[0]
        num_nodes = x.shape[0] // batch_size
        
        # Reshape node features: [batch_size, num_nodes * node_feature_dim]
        x_flat = x.view(batch_size, -1)
        
        # Concatenate with global features
        features = torch.cat([x_flat, global_features], dim=1)
        
        # Forward through MLP
        out = self.mlp(features)
        
        return out


def create_mlp_model(config: dict) -> MLPPredictor:
    """Create MLP model from config."""
    model = MLPPredictor(
        node_feature_dim=config.get('node_feature_dim', 3),
        global_feature_dim=config.get('global_feature_dim', 8),
        num_nodes=config.get('num_nodes', 8),
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        dropout=config.get('dropout', 0.2)
    )
    return model


if __name__ == '__main__':
    # Test
    model = MLPPredictor()
    
    batch_size = 32
    num_nodes = 8
    
    x = torch.randn(batch_size * num_nodes, 3)
    edge_index = torch.randint(0, batch_size * num_nodes, (2, 100))
    global_features = torch.randn(batch_size, 8)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    out = model(x, edge_index, global_features, batch)
    
    print(f"\nTest forward pass:")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")