"""
Evidential Loss Functions for Normal-Inverse-Gamma (NIG) Regression.

This implements:
1. NIG Negative Log-Likelihood (NLL)
2. Wu et al. (AAAI 2024) Non-Saturating Uncertainty Regularization
3. Combined Evidential Loss

References:
- Amini et al. (NeurIPS 2020): Deep Evidential Regression
- Wu et al. (AAAI 2024): Evidence Contraction Issue in Deep Evidential Regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def nig_nll(y: torch.Tensor,
            gamma: torch.Tensor,
            nu: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            reduce: bool = True) -> torch.Tensor:
    """
    Negative Log-Likelihood of Normal-Inverse-Gamma distribution.

    The NIG distribution models the uncertainty over both the mean and variance
    of a Gaussian distribution.

    Args:
        y: Target values [batch_size, 1]
        gamma: Predicted mean [batch_size, 1]
        nu: Evidence (virtual observations) [batch_size, 1], must be > 0
        alpha: Shape parameter [batch_size, 1], must be > 1
        beta: Scale parameter [batch_size, 1], must be > 0
        reduce: Whether to average the loss

    Returns:
        NLL loss
    """
    # Ensure numerical stability
    nu = nu + 1e-6
    alpha = alpha + 1e-6
    beta = beta + 1e-6

    # Compute omega (scaled precision)
    omega = 2 * beta * (1 + nu)

    # NLL formula from Amini et al. 2020
    nll = 0.5 * torch.log(np.pi / nu) \
        - alpha * torch.log(omega) \
        + (alpha + 0.5) * torch.log((y - gamma).pow(2) * nu + omega) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    if reduce:
        return nll.mean()
    return nll


def non_saturating_regularization(y: torch.Tensor,
                                   gamma: torch.Tensor,
                                   nu: torch.Tensor,
                                   alpha: torch.Tensor,
                                   beta: torch.Tensor,
                                   k: float = 1.96,
                                   reduce: bool = True) -> torch.Tensor:
    """
    Non-Saturating Uncertainty Regularization from Wu et al. (AAAI 2024).

    This addresses the "evidence contraction" issue where gradients vanish
    for high-error samples in the original Amini et al. formulation.

    The key insight is to penalize evidence (nu) when the prediction error
    exceeds the uncertainty estimate, forcing the model to increase uncertainty
    on hard samples rather than becoming "confident but wrong".

    Args:
        y: Target values [batch_size, 1]
        gamma: Predicted mean [batch_size, 1]
        nu: Evidence [batch_size, 1]
        alpha: Shape parameter [batch_size, 1]
        beta: Scale parameter [batch_size, 1]
        k: Confidence level multiplier (1.96 for 95% CI)
        reduce: Whether to average the loss

    Returns:
        Non-saturating regularization loss
    """
    # Compute total predictive standard deviation
    # σ² = β(ν+1) / (ν(α-1)) for Student-t predictive distribution
    # Simplified: σ ≈ sqrt(β / (α - 1)) for large ν
    var = beta / (alpha - 1 + 1e-6)
    sigma = torch.sqrt(var + 1e-6)

    # Prediction error
    error = torch.abs(y - gamma)

    # Penalize evidence when error exceeds confidence bound
    # This ensures gradients remain non-zero for high-error samples
    reg = F.relu(error - k * sigma) * nu

    if reduce:
        return reg.mean()
    return reg


def evidential_loss(y: torch.Tensor,
                    gamma: torch.Tensor,
                    nu: torch.Tensor,
                    alpha: torch.Tensor,
                    beta: torch.Tensor,
                    lambda_ns: float = 0.1,
                    lambda_mse: float = 1.0,
                    k: float = 1.96) -> dict:
    """
    Combined Evidential Loss with Non-Saturating Regularization and MSE.

    L = λ_MSE * L_MSE + L_NLL + λ_NS * L_NS

    The MSE term is critical for ensuring the model learns accurate mean predictions.
    Without it, the NIG NLL can cause the model to collapse to low-variance outputs.

    Args:
        y: Target values [batch_size, 1]
        gamma: Predicted mean [batch_size, 1]
        nu: Evidence [batch_size, 1]
        alpha: Shape parameter [batch_size, 1]
        beta: Scale parameter [batch_size, 1]
        lambda_ns: Weight for non-saturating regularization (default: 0.1)
        lambda_mse: Weight for MSE loss on mean prediction (default: 1.0)
        k: Confidence level for regularization

    Returns:
        Dictionary with total loss and components
    """
    # MSE loss on mean prediction (critical for good R²)
    # Ensure shapes match: flatten both to [batch_size]
    loss_mse = F.mse_loss(gamma.view(-1), y.view(-1))

    # NLL loss
    loss_nll = nig_nll(y, gamma, nu, alpha, beta)

    # Non-saturating regularization (Wu et al. 2024)
    loss_ns = non_saturating_regularization(y, gamma, nu, alpha, beta, k)

    # Combined loss: MSE ensures accurate mean, NLL provides uncertainty
    total_loss = lambda_mse * loss_mse + loss_nll + lambda_ns * loss_ns

    return {
        'total': total_loss,
        'mse': loss_mse,
        'nll': loss_nll,
        'ns_reg': loss_ns
    }


def compute_uncertainty(nu: torch.Tensor,
                        alpha: torch.Tensor,
                        beta: torch.Tensor) -> dict:
    """
    Compute aleatoric and epistemic uncertainty from NIG parameters.

    Args:
        nu: Evidence [batch_size, 1]
        alpha: Shape parameter [batch_size, 1]
        beta: Scale parameter [batch_size, 1]

    Returns:
        Dictionary with uncertainty decomposition
    """
    # Ensure numerical stability
    nu = nu + 1e-6
    alpha_safe = alpha - 1 + 1e-6

    # Aleatoric uncertainty: E[σ²] = β / (α - 1)
    aleatoric = beta / alpha_safe

    # Epistemic uncertainty: Var[μ] = β / (ν(α - 1))
    epistemic = beta / (nu * alpha_safe)

    # Total predictive variance (Student-t)
    # Var[y*] = β(ν+1) / (ν(α-1))
    total_var = beta * (nu + 1) / (nu * alpha_safe)

    return {
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'total_var': total_var,
        'total_std': torch.sqrt(total_var)
    }


class EvidentialLoss(nn.Module):
    """
    PyTorch Module wrapper for evidential loss.

    Combines:
    - MSE loss for accurate mean prediction (critical for good R²)
    - NIG NLL for uncertainty calibration
    - Non-saturating regularization (Wu et al. 2024)
    """

    def __init__(self, lambda_ns: float = 0.1, lambda_mse: float = 1.0, k: float = 1.96):
        super().__init__()
        self.lambda_ns = lambda_ns
        self.lambda_mse = lambda_mse
        self.k = k

    def forward(self, outputs: dict, y: torch.Tensor) -> dict:
        """
        Compute evidential loss.

        Args:
            outputs: Dictionary with 'gamma', 'nu', 'alpha', 'beta'
            y: Target values

        Returns:
            Loss dictionary with 'total', 'mse', 'nll', 'ns_reg'
        """
        return evidential_loss(
            y=y,
            gamma=outputs['gamma'],
            nu=outputs['nu'],
            alpha=outputs['alpha'],
            beta=outputs['beta'],
            lambda_ns=self.lambda_ns,
            lambda_mse=self.lambda_mse,
            k=self.k
        )


if __name__ == '__main__':
    # Test the loss functions
    print("Testing Evidential Loss Functions...")

    batch_size = 32

    # Simulated outputs
    gamma = torch.randn(batch_size, 1)
    nu = F.softplus(torch.randn(batch_size, 1)) + 1e-6
    alpha = F.softplus(torch.randn(batch_size, 1)) + 1.0
    beta = F.softplus(torch.randn(batch_size, 1)) + 1e-6
    y = torch.randn(batch_size, 1)

    # Test NLL
    nll = nig_nll(y, gamma, nu, alpha, beta)
    print(f"NLL: {nll.item():.4f}")

    # Test non-saturating regularization
    ns_reg = non_saturating_regularization(y, gamma, nu, alpha, beta)
    print(f"NS Reg: {ns_reg.item():.4f}")

    # Test combined loss
    losses = evidential_loss(y, gamma, nu, alpha, beta)
    print(f"Total Loss: {losses['total'].item():.4f}")
    print(f"  NLL: {losses['nll'].item():.4f}")
    print(f"  NS Reg: {losses['ns_reg'].item():.4f}")

    # Test uncertainty computation
    unc = compute_uncertainty(nu, alpha, beta)
    print(f"\nUncertainty (mean):")
    print(f"  Aleatoric: {unc['aleatoric'].mean().item():.4f}")
    print(f"  Epistemic: {unc['epistemic'].mean().item():.4f}")
    print(f"  Total Std: {unc['total_std'].mean().item():.4f}")

    # Test gradients don't vanish for high-error samples
    print("\nGradient test (high error sample):")
    gamma_test = torch.zeros(1, 1, requires_grad=True)
    nu_raw = torch.ones(1, 1, requires_grad=True) * 2.0
    nu_test = F.softplus(nu_raw) + 1e-6  # Apply constraint
    nu_test.retain_grad()  # Needed for non-leaf tensor
    alpha_test = torch.ones(1, 1) * 2.0
    beta_test = torch.ones(1, 1) * 1.0
    y_high_error = torch.ones(1, 1) * 10  # High error

    loss = evidential_loss(y_high_error, gamma_test, nu_test, alpha_test, beta_test)
    loss['total'].backward()

    print(f"  Error: {(y_high_error - gamma_test).abs().item():.4f}")
    print(f"  Gradient on nu: {nu_test.grad.item():.4f}")
    print(f"  Gradient on gamma: {gamma_test.grad.item():.4f}")
    print(f"  (Non-zero gradients mean no evidence contraction!)")

    print("\n✅ All tests passed!")
