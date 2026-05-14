"""
GraphPerf-RT — Section 5 (NIG evidential head): model definition.

Wraps the GAT backbone from §4 with an evidential output head that produces
the four parameters (γ, ν, α, β) of a Normal-Inverse-Gamma distribution.
The single forward pass yields **calibrated aleatoric and epistemic
uncertainty** without sampling or ensembling — a key contribution of the
paper (see §5.2 and Table 3).

Used by:
    - source/nig/train.py            single-config training
    - source/nig/hyperparam_configs  → slurm_array.sh                 (the 48-config sweep)
    - source/figures/plot_figures.py (subcommands nig-*)
    - source/rl/on_device/server_world_model.py (uncertainty gating)

References
----------
    Amini et al.  (NeurIPS 2020) — Deep Evidential Regression.
    Wu    et al.  (AAAI 2024)    — Non-saturating regularizer (we adopt this
                                   variant; see loss.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from typing import Dict, Optional


class EvidentialHead(nn.Module):
    """
    Evidential output head that produces NIG distribution parameters.

    Outputs:
        gamma (γ): Predicted mean
        nu (ν): Evidence (virtual observations), constrained > 0
        alpha (α): Shape parameter, constrained > 1
        beta (β): Scale parameter, constrained > 0

    Key design choices for stable training:
    1. Direct gamma prediction from input (no excessive compression)
    2. Proper weight initialization scaled for normalized targets
    3. Separate pathways for mean vs uncertainty parameters
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Direct pathway for gamma (mean prediction) - critical for good R²
        # Simpler path allows faster learning of the mean
        self.gamma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Uncertainty pathway for nu, alpha, beta
        # These can be learned more slowly
        self.uncertainty_shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.nu_head = nn.Linear(hidden_dim // 2, 1)     # Evidence > 0
        self.alpha_head = nn.Linear(hidden_dim // 2, 1)  # Shape > 1
        self.beta_head = nn.Linear(hidden_dim // 2, 1)   # Scale > 0

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable evidential training."""
        # Initialize gamma network with larger variance for better initial spread
        for module in self.gamma_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        # Initialize uncertainty heads with small weights
        # This starts with low confidence, which is appropriate
        for head in [self.nu_head, self.alpha_head, self.beta_head]:
            nn.init.xavier_normal_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing NIG parameters.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Dictionary with gamma, nu, alpha, beta tensors
        """
        # Gamma: direct prediction for mean (unbounded)
        gamma = self.gamma_net(x)

        # Uncertainty parameters from shared features
        unc_features = self.uncertainty_shared(x)

        # Nu: evidence, must be > 0 (softplus ensures positivity)
        # Higher nu = more confident in mean estimate
        nu = F.softplus(self.nu_head(unc_features)) + 1e-4

        # Alpha: shape parameter, must be > 1
        # Controls the shape of the uncertainty distribution
        alpha = F.softplus(self.alpha_head(unc_features)) + 1.0 + 1e-4

        # Beta: scale parameter, must be > 0
        # Related to aleatoric uncertainty
        beta = F.softplus(self.beta_head(unc_features)) + 1e-4

        return {
            'gamma': gamma,
            'nu': nu,
            'alpha': alpha,
            'beta': beta
        }


class GATEvidential(nn.Module):
    """
    Graph Attention Network with Evidential Output for Uncertainty Quantification.

    This model extends the standard GAT architecture with NIG output heads
    for calibrated uncertainty estimation.
    """

    def __init__(self,
                 node_feature_dim: int = 3,
                 global_feature_dim: int = 8,
                 hidden_dim: int = 128,
                 num_gat_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 pooling: str = 'both'):
        """
        Initialize GAT Evidential model.

        Args:
            node_feature_dim: Dimension of node features (default: 3 for core features)
            global_feature_dim: Dimension of global features
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
        self.dropout = dropout

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

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

        # Fusion layer
        fusion_input_dim = pooled_dim + hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Evidential head
        self.evidential_head = EvidentialHead(hidden_dim, hidden_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                global_features: torch.Tensor,
                batch: torch.Tensor,
                return_embedding: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            global_features: Global features [batch_size, global_feature_dim]
            batch: Batch assignment [num_nodes]
            return_embedding: Whether to return intermediate embedding

        Returns:
            Dictionary with NIG parameters (gamma, nu, alpha, beta)
            and optionally the embedding
        """
        # Encode node features
        x = self.node_encoder(x)

        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

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

        # Apply fusion layers
        embedding = self.fusion(fused)

        # Get NIG parameters from evidential head
        outputs = self.evidential_head(embedding)

        if return_embedding:
            outputs['embedding'] = embedding

        return outputs

    def predict_with_uncertainty(self,
                                  x: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  global_features: torch.Tensor,
                                  batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make prediction with uncertainty estimates.

        Returns:
            Dictionary with prediction, aleatoric, epistemic, and total uncertainty
        """
        outputs = self.forward(x, edge_index, global_features, batch)

        gamma = outputs['gamma']
        nu = outputs['nu']
        alpha = outputs['alpha']
        beta = outputs['beta']

        # Compute uncertainties
        alpha_safe = alpha - 1 + 1e-6

        # Aleatoric: E[σ²] = β / (α - 1)
        aleatoric_var = beta / alpha_safe

        # Epistemic: Var[μ] = β / (ν(α - 1))
        epistemic_var = beta / (nu * alpha_safe)

        # Total predictive variance
        total_var = beta * (nu + 1) / (nu * alpha_safe)

        return {
            'prediction': gamma,
            'aleatoric_std': torch.sqrt(aleatoric_var),
            'epistemic_std': torch.sqrt(epistemic_var),
            'total_std': torch.sqrt(total_var),
            'nu': nu,
            'alpha': alpha,
            'beta': beta
        }


def create_evidential_model(config: dict) -> GATEvidential:
    """Create model from config dictionary."""
    model = GATEvidential(
        node_feature_dim=config.get('node_feature_dim', 3),
        global_feature_dim=config.get('global_feature_dim', 8),
        hidden_dim=config.get('hidden_dim', 128),
        num_gat_layers=config.get('num_gat_layers', 3),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.2),
        pooling=config.get('pooling', 'both')
    )
    return model


if __name__ == '__main__':
    # Test model
    print("Testing GATEvidential...")

    model = GATEvidential(
        node_feature_dim=3,
        global_feature_dim=69,  # Match actual dataset
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=4,
        dropout=0.2,
        pooling='both'
    )

    print(f"\nModel architecture:")
    print(model)

    # Test forward pass
    batch_size = 32
    num_cores = 8
    num_nodes = batch_size * num_cores

    x = torch.randn(num_nodes, 3)
    # Create fully connected edges within each graph
    edges = []
    for b in range(batch_size):
        offset = b * num_cores
        for i in range(num_cores):
            for j in range(num_cores):
                if i != j:
                    edges.append([offset + i, offset + j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    global_features = torch.randn(batch_size, 69)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_cores)

    outputs = model(x, edge_index, global_features, batch)

    print(f"\nForward pass test:")
    print(f"  Input: x={x.shape}, edge_index={edge_index.shape}")
    print(f"  Global: {global_features.shape}, batch={batch.shape}")
    print(f"  Output gamma: {outputs['gamma'].shape}")
    print(f"  Output nu: {outputs['nu'].shape}")
    print(f"  Output alpha: {outputs['alpha'].shape}")
    print(f"  Output beta: {outputs['beta'].shape}")

    # Test predict_with_uncertainty
    model.eval()
    with torch.no_grad():
        preds = model.predict_with_uncertainty(x, edge_index, global_features, batch)

    print(f"\nUncertainty estimates:")
    print(f"  Prediction: {preds['prediction'].shape}")
    print(f"  Aleatoric std: {preds['aleatoric_std'].mean().item():.4f}")
    print(f"  Epistemic std: {preds['epistemic_std'].mean().item():.4f}")
    print(f"  Total std: {preds['total_std'].mean().item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n✅ Model test passed!")
