#!/usr/bin/env python3
"""
Training script for GAT Evidential model with NIG uncertainty.

This script trains the GATEvidential model on the performance prediction dataset
and evaluates calibration quality.

Uses V2 dataset format with one-hot encoded categorical features:
- bench_0 to bench_41 (42 benchmarks)
- plat_0, plat_1 (2 platforms)
- mode_0, mode_1 (2 run modes)
- var_0, var_1, var_2 (3 variants)

References:
- Amini et al. (NeurIPS 2020): Deep Evidential Regression
- Wu et al. (AAAI 2024): Non-Saturating Uncertainty Regularization
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Path resolution. This file lives at  source/nig/train.py
#   - sibling NIG modules (model.py, loss.py) are co-located.
#   - the V2 dataset class is shared with the GAT pipeline and lives at
#     source/gat/src/data/dataset.py.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SOURCE_ROOT / 'gat' / 'src' / 'data'))
sys.path.insert(0, str(Path(__file__).parent))

from model import GATEvidential, create_evidential_model     # noqa: E402
from loss import EvidentialLoss, compute_uncertainty         # noqa: E402
from dataset import CoreGraphDatasetV2, collate_fn           # noqa: E402


class EvidentialTrainer:
    """Training pipeline for GAT Evidential model."""

    def __init__(self, config: dict, experiment_name: str, device: str = 'auto'):
        self.config = config
        self.experiment_name = experiment_name

        # Setup device based on argument
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        elif device == 'cuda':
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # Create experiment directory
        self.exp_dir = Path(__file__).parent / 'experiments' / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize tracking
        self.best_val_loss = float('inf')
        self.best_val_r2 = -float('inf')
        self.patience_counter = 0
        self.start_epoch = 0  # Will be updated if resuming
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_nll': [], 'val_nll': [],
            'train_r2': [], 'val_r2': [],
            'val_mse': [], 'val_mae': [],
            'val_ece': []
        }

    def setup_data(self):
        """Setup data loaders using V2 dataset with one-hot encoded features."""
        print("\nSetting up data loaders (V2 dataset format)...")

        base_path = Path(__file__).parent.parent / 'GAT_EFG'

        # Resolve paths
        train_path = base_path / self.config['data']['train_path']
        val_path = base_path / self.config['data']['val_path']
        test_path = base_path / self.config['data']['test_path']

        metadata_path = None
        if self.config['data'].get('metadata_path'):
            metadata_path = base_path / self.config['data']['metadata_path']
            if not metadata_path.exists():
                print(f"  Warning: metadata file not found at {metadata_path}")
                metadata_path = None

        print(f"  Train data: {train_path}")
        print(f"  Val data:   {val_path}")
        print(f"  Test data:  {test_path}")

        # Create V2 datasets with one-hot encoded features
        self.train_dataset = CoreGraphDatasetV2(
            csv_path=str(train_path),
            metadata_path=str(metadata_path) if metadata_path else None,
            target_column=self.config['data']['target_column'],
            use_cfg_features=False  # CFG features not in v2 data
        )

        self.val_dataset = CoreGraphDatasetV2(
            csv_path=str(val_path),
            metadata_path=str(metadata_path) if metadata_path else None,
            target_column=self.config['data']['target_column'],
            use_cfg_features=False
        )

        self.test_dataset = CoreGraphDatasetV2(
            csv_path=str(test_path),
            metadata_path=str(metadata_path) if metadata_path else None,
            target_column=self.config['data']['target_column'],
            use_cfg_features=False
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 for debugging
            collate_fn=collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        # Get feature dimensions
        self.feature_dims = self.train_dataset.get_feature_dims()

        print(f"\nDataLoaders created:")
        print(f"  Train: {len(self.train_dataset)} samples, {len(self.train_loader)} batches")
        print(f"  Val:   {len(self.val_dataset)} samples, {len(self.val_loader)} batches")
        print(f"  Test:  {len(self.test_dataset)} samples, {len(self.test_loader)} batches")
        print(f"  Feature dims: node={self.feature_dims[0]}, global={self.feature_dims[1]}")

    def setup_model(self):
        """Setup model, optimizer, and loss function."""
        print("\nSetting up model...")

        # Update config with actual feature dimensions
        self.config['model']['node_feature_dim'] = self.feature_dims[0]
        self.config['model']['global_feature_dim'] = self.feature_dims[1]

        # Create model
        self.model = create_evidential_model(self.config['model']).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Evidential loss with MSE component for stable training
        self.criterion = EvidentialLoss(
            lambda_ns=self.config['training'].get('lambda_ns', 0.1),
            lambda_mse=self.config['training'].get('lambda_mse', 1.0),
            k=self.config['training'].get('confidence_k', 1.96)
        )

        # Try to resume from checkpoint if exists
        self._try_resume_from_checkpoint()

    def _try_resume_from_checkpoint(self):
        """
        Automatically resume from latest checkpoint if it exists.
        This allows training to continue after job interruption.
        """
        checkpoint_path = self.exp_dir / 'latest.pth'

        if not checkpoint_path.exists():
            print("No checkpoint found. Starting training from scratch.")
            return

        print(f"\n{'='*60}")
        print("RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)

            # Restore model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Restore optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore scheduler state
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Restore tracking variables
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
            self.start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch

            # Restore history
            if 'history' in checkpoint:
                self.history = checkpoint['history']

            # Restore patience counter if saved
            self.patience_counter = checkpoint.get('patience_counter', 0)

            print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown') + 1}")
            print(f"  Best Val R²: {self.best_val_r2:.4f}")
            print(f"  Best Val Loss: {self.best_val_loss:.4f}")
            print(f"  Resuming from epoch {self.start_epoch + 1}")
            print(f"  Patience counter: {self.patience_counter}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"  WARNING: Failed to load checkpoint: {e}")
            print("  Starting training from scratch.")
            self.start_epoch = 0

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_nll = 0
        total_mse = 0
        total_ns_reg = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(
                batch.x,
                batch.edge_index,
                batch.global_features,
                batch.batch
            )

            # Compute evidential loss
            losses = self.criterion(outputs, batch.y)
            loss = losses['total']

            # Backward
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_nll += losses['nll'].item()
            total_mse += losses['mse'].item()
            total_ns_reg += losses['ns_reg'].item()

            # Collect predictions
            all_preds.append(outputs['gamma'].detach())
            all_targets.append(batch.y.detach())

            pbar.set_postfix({'loss': loss.item(), 'mse': losses['mse'].item()})

        # Compute metrics - flatten to avoid shape broadcasting issues
        all_preds = torch.cat(all_preds).cpu().numpy().flatten()
        all_targets = torch.cat(all_targets).cpu().numpy().flatten()
        r2 = self._compute_r2(all_targets, all_preds)

        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'nll': total_nll / n_batches,
            'mse': total_mse / n_batches,
            'ns_reg': total_ns_reg / n_batches,
            'r2': r2,
            'pred_mean': float(np.mean(all_preds)),
            'pred_std': float(np.std(all_preds)),
            'target_mean': float(np.mean(all_targets)),
            'target_std': float(np.std(all_targets))
        }

    @torch.no_grad()
    def validate(self, loader=None) -> dict:
        """Validate model."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        total_loss = 0
        total_nll = 0
        all_preds = []
        all_targets = []
        all_stds = []

        for batch in loader:
            batch = batch.to(self.device)

            outputs = self.model(
                batch.x,
                batch.edge_index,
                batch.global_features,
                batch.batch
            )

            losses = self.criterion(outputs, batch.y)
            total_loss += losses['total'].item()
            total_nll += losses['nll'].item()

            # Compute uncertainty
            unc = compute_uncertainty(outputs['nu'], outputs['alpha'], outputs['beta'])

            all_preds.append(outputs['gamma'].cpu())
            all_targets.append(batch.y.cpu())
            all_stds.append(unc['total_std'].cpu())

        # Aggregate and flatten to avoid shape broadcasting issues
        all_preds = torch.cat(all_preds).numpy().flatten()
        all_targets = torch.cat(all_targets).numpy().flatten()
        all_stds = torch.cat(all_stds).numpy().flatten()

        # Compute metrics (arrays are now guaranteed to be 1D)
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))
        r2 = self._compute_r2(all_targets, all_preds)
        ece = self._compute_ece(all_targets, all_preds, all_stds)

        # Additional metrics for paper
        mape = self._compute_mape(all_targets, all_preds)
        spearman = self._compute_spearman(all_targets, all_preds)
        mce = self._compute_mce(all_targets, all_preds, all_stds)

        return {
            'loss': total_loss / len(loader),
            'nll': total_nll / len(loader),
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'ece': ece,
            'mce': mce,
            'spearman': spearman,
            'predictions': all_preds,
            'targets': all_targets,
            'stds': all_stds
        }

    def _compute_r2(self, y_true, y_pred):
        """Compute R² score."""
        # CRITICAL: Flatten arrays to avoid broadcasting issues
        # If y_true is [N] and y_pred is [N,1], subtraction creates [N,N] matrix!
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    def _compute_mape(self, y_true, y_pred):
        """Compute Mean Absolute Percentage Error."""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        # Avoid division by zero
        mask = np.abs(y_true) > 1e-8
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def _compute_spearman(self, y_true, y_pred):
        """Compute Spearman rank correlation coefficient."""
        from scipy import stats
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        corr, _ = stats.spearmanr(y_true, y_pred)
        return corr

    def _compute_mce(self, y_true, y_pred, y_std, n_bins=10):
        """
        Compute Maximum Calibration Error.

        MCE is the maximum absolute difference between expected and actual coverage
        across all confidence levels.
        """
        from scipy import stats

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        y_std = y_std.flatten()

        conf_levels = np.linspace(0.1, 0.9, n_bins)
        max_error = 0

        for conf in conf_levels:
            z = stats.norm.ppf((1 + conf) / 2)
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            in_interval = (y_true >= lower) & (y_true <= upper)
            actual_coverage = np.mean(in_interval)
            error = np.abs(actual_coverage - conf)
            max_error = max(max_error, error)

        return max_error

    def _compute_ece(self, y_true, y_pred, y_std, n_bins=10):
        """
        Compute Expected Calibration Error for regression.

        For each confidence level, check if the actual coverage matches expected.
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        y_std = y_std.flatten()

        # Confidence levels to check
        conf_levels = np.linspace(0.1, 0.9, n_bins)
        ece = 0

        for conf in conf_levels:
            # z-score for this confidence level (two-sided)
            from scipy import stats
            z = stats.norm.ppf((1 + conf) / 2)

            # Predicted interval
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std

            # Actual coverage
            in_interval = (y_true >= lower) & (y_true <= upper)
            actual_coverage = np.mean(in_interval)

            # Calibration error
            ece += np.abs(actual_coverage - conf)

        return ece / len(conf_levels)

    def _compute_picp_mpiw(self, y_true, y_pred, y_std, confidence=0.95):
        """
        Compute PICP (Prediction Interval Coverage Probability) and
        MPIW (Mean Prediction Interval Width) for the paper.

        These are critical calibration metrics for uncertainty quantification.

        Args:
            y_true: Ground truth values
            y_pred: Predicted means
            y_std: Predicted standard deviations
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Dictionary with PICP, MPIW, and normalized MPIW (PINAW)
        """
        from scipy import stats

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        y_std = y_std.flatten()

        # z-score for confidence level (two-sided)
        z = stats.norm.ppf((1 + confidence) / 2)

        # Prediction intervals
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std

        # PICP: Prediction Interval Coverage Probability
        # Fraction of true values that fall within the predicted interval
        in_interval = (y_true >= lower) & (y_true <= upper)
        picp = np.mean(in_interval)

        # MPIW: Mean Prediction Interval Width
        interval_widths = upper - lower
        mpiw = np.mean(interval_widths)

        # PINAW: Prediction Interval Normalized Average Width
        # Normalized by the range of true values
        y_range = y_true.max() - y_true.min()
        pinaw = mpiw / y_range if y_range > 0 else 0.0

        # CWC: Coverage Width-based Criterion (penalizes under-coverage)
        # Lower is better - balances coverage and width
        eta = 50  # Penalty factor for under-coverage
        gamma = 1 if picp < confidence else 0
        cwc = pinaw * (1 + gamma * np.exp(-eta * (picp - confidence)))

        return {
            'picp': picp,
            'mpiw': mpiw,
            'pinaw': pinaw,
            'cwc': cwc,
            'confidence': confidence
        }

    def _compute_uncertainty_decomposition(self, loader):
        """
        Compute aleatoric and epistemic uncertainty decomposition.

        This is important for the paper to show that the model captures
        both data noise (aleatoric) and model uncertainty (epistemic).
        Also collects NIG parameters (nu, alpha, beta) statistics.
        """
        self.model.eval()
        all_aleatoric = []
        all_epistemic = []
        all_total = []
        all_nu = []
        all_alpha = []
        all_beta = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs = self.model(
                    batch.x, batch.edge_index,
                    batch.global_features, batch.batch
                )
                unc = compute_uncertainty(
                    outputs['nu'], outputs['alpha'], outputs['beta']
                )
                all_aleatoric.append(unc['aleatoric'].cpu())
                all_epistemic.append(unc['epistemic'].cpu())
                all_total.append(unc['total_var'].cpu())
                # Collect NIG parameters
                all_nu.append(outputs['nu'].cpu())
                all_alpha.append(outputs['alpha'].cpu())
                all_beta.append(outputs['beta'].cpu())

        aleatoric = torch.cat(all_aleatoric).numpy().flatten()
        epistemic = torch.cat(all_epistemic).numpy().flatten()
        total = torch.cat(all_total).numpy().flatten()
        nu = torch.cat(all_nu).numpy().flatten()
        alpha = torch.cat(all_alpha).numpy().flatten()
        beta = torch.cat(all_beta).numpy().flatten()

        return {
            'aleatoric_mean': float(np.mean(aleatoric)),
            'aleatoric_std': float(np.std(aleatoric)),
            'epistemic_mean': float(np.mean(epistemic)),
            'epistemic_std': float(np.std(epistemic)),
            'total_var_mean': float(np.mean(total)),
            'total_var_std': float(np.std(total)),
            'epistemic_ratio': float(np.mean(epistemic) / (np.mean(aleatoric) + np.mean(epistemic) + 1e-8)),
            # NIG parameter statistics for paper
            'nu_mean': float(np.mean(nu)),
            'nu_std': float(np.std(nu)),
            'nu_min': float(np.min(nu)),
            'nu_max': float(np.max(nu)),
            'alpha_mean': float(np.mean(alpha)),
            'alpha_std': float(np.std(alpha)),
            'alpha_min': float(np.min(alpha)),
            'alpha_max': float(np.max(alpha)),
            'beta_mean': float(np.mean(beta)),
            'beta_std': float(np.std(beta)),
            'beta_min': float(np.min(beta)),
            'beta_max': float(np.max(beta)),
            # Raw arrays for detailed analysis
            'nu_array': nu,
            'alpha_array': alpha,
            'beta_array': beta
        }

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with all state for resume."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_r2': self.best_val_r2,
            'patience_counter': self.patience_counter,
            'config': self.config,
            'history': self.history
        }

        torch.save(checkpoint, self.exp_dir / 'latest.pth')
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best.pth')

    def plot_results(self, val_results: dict):
        """Plot training curves and calibration."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # R² curves
        axes[0, 1].plot(self.history['train_r2'], label='Train')
        axes[0, 1].plot(self.history['val_r2'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_title('R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Predictions vs Targets
        axes[0, 2].scatter(val_results['targets'], val_results['predictions'],
                          alpha=0.5, s=5)
        axes[0, 2].plot([val_results['targets'].min(), val_results['targets'].max()],
                       [val_results['targets'].min(), val_results['targets'].max()],
                       'r--', label='Perfect')
        axes[0, 2].set_xlabel('True Values')
        axes[0, 2].set_ylabel('Predictions')
        axes[0, 2].set_title(f'Predictions (R²={val_results["r2"]:.4f})')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Uncertainty vs Error
        errors = np.abs(val_results['predictions'].flatten() - val_results['targets'].flatten())
        axes[1, 0].scatter(val_results['stds'].flatten(), errors, alpha=0.3, s=5)
        axes[1, 0].set_xlabel('Predicted Uncertainty (std)')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].set_title('Uncertainty vs Error')
        axes[1, 0].grid(True)

        # Calibration plot
        from scipy import stats
        conf_levels = np.linspace(0.1, 0.95, 20)
        actual_coverages = []
        for conf in conf_levels:
            z = stats.norm.ppf((1 + conf) / 2)
            lower = val_results['predictions'].flatten() - z * val_results['stds'].flatten()
            upper = val_results['predictions'].flatten() + z * val_results['stds'].flatten()
            in_interval = (val_results['targets'].flatten() >= lower) & \
                         (val_results['targets'].flatten() <= upper)
            actual_coverages.append(np.mean(in_interval))

        axes[1, 1].plot(conf_levels, actual_coverages, 'b-o', label='Actual')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect')
        axes[1, 1].set_xlabel('Expected Coverage')
        axes[1, 1].set_ylabel('Actual Coverage')
        axes[1, 1].set_title(f'Calibration (ECE={val_results["ece"]:.4f})')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # ECE over training
        if self.history['val_ece']:
            axes[1, 2].plot(self.history['val_ece'])
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('ECE')
            axes[1, 2].set_title('Calibration Error Over Training')
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(self.exp_dir / 'training_results.png', dpi=150, bbox_inches='tight')
        plt.close()

    def train(self):
        """Main training loop with automatic resume support."""
        total_epochs = self.config['training']['num_epochs']

        # Check if training already completed
        if self.start_epoch >= total_epochs:
            print(f"\n{'='*60}")
            print(f"Training already completed ({self.start_epoch} epochs)")
            print(f"Skipping to final evaluation...")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            if self.start_epoch > 0:
                print(f"Resuming training: {self.experiment_name}")
                print(f"  From epoch {self.start_epoch + 1} to {total_epochs}")
            else:
                print(f"Starting training: {self.experiment_name}")
            print(f"{'='*60}\n")

            for epoch in range(self.start_epoch, total_epochs):
                # Train
                train_metrics = self.train_epoch()
                self.history['train_loss'].append(float(train_metrics['loss']))
                self.history['train_nll'].append(float(train_metrics['nll']))
                self.history['train_r2'].append(float(train_metrics['r2']))

                # Validate
                val_metrics = self.validate()
                self.history['val_loss'].append(float(val_metrics['loss']))
                self.history['val_nll'].append(float(val_metrics['nll']))
                self.history['val_r2'].append(float(val_metrics['r2']))
                self.history['val_mse'].append(float(val_metrics['mse']))
                self.history['val_mae'].append(float(val_metrics['mae']))
                self.history['val_ece'].append(float(val_metrics['ece']))

                # Update scheduler
                self.scheduler.step(val_metrics['loss'])
                current_lr = self.optimizer.param_groups[0]['lr']

                # Diagnostic output
                print(f"Epoch {epoch+1:03d}: "
                      f"Loss={train_metrics['loss']:.4f} (MSE={train_metrics['mse']:.4f}, NLL={train_metrics['nll']:.4f}) | "
                      f"Train R²={train_metrics['r2']:.4f}, Val R²={val_metrics['r2']:.4f}")
                print(f"         Pred: mean={train_metrics['pred_mean']:.3f}, std={train_metrics['pred_std']:.3f} | "
                      f"Target: mean={train_metrics['target_mean']:.3f}, std={train_metrics['target_std']:.3f} | "
                      f"LR={current_lr:.6f}")

                # Check for improvement (using R² as primary metric)
                is_best = val_metrics['r2'] > self.best_val_r2
                if is_best:
                    self.best_val_r2 = val_metrics['r2']
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    print(f"  → New best R²: {val_metrics['r2']:.4f}")
                else:
                    self.patience_counter += 1

                # Save checkpoint
                self.save_checkpoint(epoch, is_best)

                # Early stopping
                if self.patience_counter >= self.config['training']['patience']:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Final evaluation on test set
        print(f"\n{'='*60}")
        print("Final Evaluation on Test Set")
        print(f"{'='*60}")

        # Load best model (weights_only=False needed for PyTorch 2.6+ compatibility)
        checkpoint = torch.load(self.exp_dir / 'best.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = self.validate(self.test_loader)

        # Compute additional metrics for the paper
        picp_mpiw = self._compute_picp_mpiw(
            test_metrics['targets'],
            test_metrics['predictions'],
            test_metrics['stds'],
            confidence=0.95
        )

        # Compute uncertainty decomposition
        unc_decomp = self._compute_uncertainty_decomposition(self.test_loader)

        print(f"\n" + "="*60)
        print("TEST RESULTS (for IJCAI Paper)")
        print("="*60)
        print(f"\nPrediction Metrics (Table 1):")
        print(f"  R²:       {test_metrics['r2']:.4f}")
        print(f"  RMSE:     {np.sqrt(test_metrics['mse']):.4f}")
        print(f"  MAE:      {test_metrics['mae']:.4f}")
        print(f"  MAPE:     {test_metrics['mape']:.2f}%")
        print(f"  Spearman: {test_metrics['spearman']:.4f}")
        print(f"\nCalibration Metrics (Table 2):")
        print(f"  ECE:        {test_metrics['ece']:.4f}")
        print(f"  MCE:        {test_metrics['mce']:.4f}")
        print(f"  Reliability (PICP@95%): {picp_mpiw['picp']:.4f}")
        print(f"  Sharpness (MPIW):       {picp_mpiw['mpiw']:.4f}")
        print(f"  PINAW:    {picp_mpiw['pinaw']:.4f}")
        print(f"\nUncertainty Decomposition:")
        print(f"  Aleatoric (mean): {unc_decomp['aleatoric_mean']:.6f}")
        print(f"  Epistemic (mean): {unc_decomp['epistemic_mean']:.6f}")
        print(f"  Epistemic Ratio:  {unc_decomp['epistemic_ratio']:.4f}")
        print(f"\nNIG Parameters (γ=prediction, ν=evidence, α=shape, β=scale):")
        print(f"  ν (nu):    mean={unc_decomp['nu_mean']:.4f}, std={unc_decomp['nu_std']:.4f}, range=[{unc_decomp['nu_min']:.4f}, {unc_decomp['nu_max']:.4f}]")
        print(f"  α (alpha): mean={unc_decomp['alpha_mean']:.4f}, std={unc_decomp['alpha_std']:.4f}, range=[{unc_decomp['alpha_min']:.4f}, {unc_decomp['alpha_max']:.4f}]")
        print(f"  β (beta):  mean={unc_decomp['beta_mean']:.4f}, std={unc_decomp['beta_std']:.4f}, range=[{unc_decomp['beta_min']:.4f}, {unc_decomp['beta_max']:.4f}]")

        # Plot results
        self.plot_results(test_metrics)

        # Generate additional paper figures
        self._plot_picp_mpiw_figure(test_metrics)
        self._plot_uncertainty_decomposition(test_metrics, unc_decomp)

        # Save final results with all metrics needed for the IJCAI paper
        results = {
            # ===========================================
            # PAPER TABLE 1: Prediction Metrics
            # ===========================================
            'test_r2': float(test_metrics['r2']),
            'test_rmse': float(np.sqrt(test_metrics['mse'])),
            'test_mae': float(test_metrics['mae']),
            'test_mape': float(test_metrics['mape']),
            'test_spearman': float(test_metrics['spearman']),

            # ===========================================
            # PAPER TABLE 2: Calibration Metrics
            # ===========================================
            'test_ece': float(test_metrics['ece']),
            'test_mce': float(test_metrics['mce']),
            'test_reliability': float(picp_mpiw['picp']),  # PICP at 95%
            'test_sharpness': float(picp_mpiw['mpiw']),    # MPIW

            # Additional calibration metrics
            'test_picp_95': float(picp_mpiw['picp']),
            'test_mpiw': float(picp_mpiw['mpiw']),
            'test_pinaw': float(picp_mpiw['pinaw']),
            'test_cwc': float(picp_mpiw['cwc']),

            # ===========================================
            # Uncertainty Decomposition
            # ===========================================
            'aleatoric_uncertainty_mean': float(unc_decomp['aleatoric_mean']),
            'aleatoric_uncertainty_std': float(unc_decomp['aleatoric_std']),
            'epistemic_uncertainty_mean': float(unc_decomp['epistemic_mean']),
            'epistemic_uncertainty_std': float(unc_decomp['epistemic_std']),
            'epistemic_ratio': float(unc_decomp['epistemic_ratio']),

            # ===========================================
            # NIG Parameters (γ, ν, α, β) Statistics
            # ===========================================
            'nig_nu_mean': float(unc_decomp['nu_mean']),
            'nig_nu_std': float(unc_decomp['nu_std']),
            'nig_nu_min': float(unc_decomp['nu_min']),
            'nig_nu_max': float(unc_decomp['nu_max']),
            'nig_alpha_mean': float(unc_decomp['alpha_mean']),
            'nig_alpha_std': float(unc_decomp['alpha_std']),
            'nig_alpha_min': float(unc_decomp['alpha_min']),
            'nig_alpha_max': float(unc_decomp['alpha_max']),
            'nig_beta_mean': float(unc_decomp['beta_mean']),
            'nig_beta_std': float(unc_decomp['beta_std']),
            'nig_beta_min': float(unc_decomp['beta_min']),
            'nig_beta_max': float(unc_decomp['beta_max']),

            # ===========================================
            # Training Info
            # ===========================================
            'best_val_r2': float(self.best_val_r2),
            'best_val_loss': float(self.best_val_loss),
            'num_epochs': len(self.history['train_loss']),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save training history
        with open(self.exp_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        # Save predictions and NIG parameters for further analysis
        np.savez(
            self.exp_dir / 'test_predictions.npz',
            predictions=test_metrics['predictions'],
            targets=test_metrics['targets'],
            stds=test_metrics['stds'],
            # NIG parameters for detailed analysis
            nu=unc_decomp['nu_array'],
            alpha=unc_decomp['alpha_array'],
            beta=unc_decomp['beta_array']
        )

        print(f"\n{'='*60}")
        print(f"Results saved to: {self.exp_dir}")
        print(f"  - results.json (all metrics)")
        print(f"  - training_history.json")
        print(f"  - training_results.png")
        print(f"  - calibration_picp_mpiw.png")
        print(f"  - uncertainty_decomposition.png")
        print(f"  - test_predictions.npz")
        print(f"{'='*60}\n")

        return results

    def _plot_picp_mpiw_figure(self, test_metrics):
        """Generate PICP vs MPIW figure for the paper."""
        from scipy import stats

        y_true = test_metrics['targets'].flatten()
        y_pred = test_metrics['predictions'].flatten()
        y_std = test_metrics['stds'].flatten()

        # Compute PICP and MPIW at multiple confidence levels
        conf_levels = np.linspace(0.5, 0.99, 20)
        picps = []
        mpiwes = []

        for conf in conf_levels:
            z = stats.norm.ppf((1 + conf) / 2)
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            in_interval = (y_true >= lower) & (y_true <= upper)
            picps.append(np.mean(in_interval))
            mpiwes.append(np.mean(upper - lower))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PICP vs Expected Coverage
        axes[0].plot(conf_levels, picps, 'b-o', markersize=4, label='Actual')
        axes[0].plot([0.5, 1], [0.5, 1], 'r--', label='Perfect Calibration')
        axes[0].fill_between(conf_levels, conf_levels - 0.05, conf_levels + 0.05,
                            alpha=0.2, color='red', label='±5% tolerance')
        axes[0].set_xlabel('Expected Coverage (Confidence Level)', fontsize=12)
        axes[0].set_ylabel('PICP (Actual Coverage)', fontsize=12)
        axes[0].set_title('Calibration: PICP vs Expected', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0.5, 1.0)
        axes[0].set_ylim(0.5, 1.0)

        # MPIW vs Confidence Level
        axes[1].plot(conf_levels, mpiwes, 'g-o', markersize=4)
        axes[1].set_xlabel('Confidence Level', fontsize=12)
        axes[1].set_ylabel('MPIW (Mean Prediction Interval Width)', fontsize=12)
        axes[1].set_title('Prediction Interval Width vs Confidence', fontsize=14)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.exp_dir / 'calibration_picp_mpiw.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_uncertainty_decomposition(self, test_metrics, unc_decomp):
        """Generate uncertainty decomposition figure for the paper."""
        errors = np.abs(test_metrics['predictions'].flatten() - test_metrics['targets'].flatten())
        stds = test_metrics['stds'].flatten()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Error vs Uncertainty scatter
        axes[0].scatter(stds, errors, alpha=0.3, s=5)
        # Add trend line
        z = np.polyfit(stds, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(stds.min(), stds.max(), 100)
        axes[0].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
        axes[0].set_xlabel('Predicted Uncertainty (σ)', fontsize=12)
        axes[0].set_ylabel('Absolute Error', fontsize=12)
        axes[0].set_title('Uncertainty vs Prediction Error', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Uncertainty decomposition pie chart
        labels = ['Aleatoric', 'Epistemic']
        sizes = [unc_decomp['aleatoric_mean'],
                unc_decomp['epistemic_mean']]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0.05)
        axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
        axes[1].set_title('Uncertainty Decomposition', fontsize=14)

        # Binned calibration plot
        n_bins = 10
        bin_indices = np.digitize(stds, np.percentile(stds, np.linspace(0, 100, n_bins + 1)))
        bin_errors = [errors[bin_indices == i].mean() for i in range(1, n_bins + 1)]
        bin_stds = [stds[bin_indices == i].mean() for i in range(1, n_bins + 1)]

        axes[2].bar(range(1, n_bins + 1), bin_errors, alpha=0.7, label='Mean Error')
        axes[2].plot(range(1, n_bins + 1), bin_stds, 'ro-', linewidth=2, label='Mean Uncertainty')
        axes[2].set_xlabel('Uncertainty Bin (Low → High)', fontsize=12)
        axes[2].set_ylabel('Value', fontsize=12)
        axes[2].set_title('Error vs Uncertainty by Bin', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.exp_dir / 'uncertainty_decomposition.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train GAT Evidential model')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--lambda_ns', type=float, default=0.1,
                       help='Non-saturating regularization weight')
    parser.add_argument('--lambda_mse', type=float, default=1.0,
                       help='MSE loss weight (critical for good R²)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (auto, cpu, or cuda)')
    parser.add_argument('--eval-only', type=str, default=None,
                       help='Path to checkpoint for evaluation only (skip training)')

    args = parser.parse_args()

    # Default experiment name
    if args.experiment is None:
        args.experiment = f"nig_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Configuration
    config = {
        'model': {
            'hidden_dim': args.hidden_dim,
            'num_gat_layers': 3,
            'num_heads': 4,
            'dropout': 0.2,
            'pooling': 'both'
        },
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': 1e-4,
            'patience': 15,
            'lambda_ns': args.lambda_ns,
            'lambda_mse': args.lambda_mse,
            'confidence_k': 1.96
        },
        'data': {
            # V2 dataset with one-hot encoded features (32,928 training samples)
            'train_path': 'data/splits/v2/train_log.csv',
            'val_path': 'data/splits/v2/val_log.csv',
            'test_path': 'data/splits/v2/test_log.csv',
            'metadata_path': 'data/processed/v2/merged_metadata_scaler.json',
            'target_column': 'time_elapsed_log'
        }
    }

    # Create trainer
    trainer = EvidentialTrainer(config, args.experiment, device=args.device)

    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()

    # Eval-only mode: load checkpoint and run final evaluation
    if args.eval_only:
        print(f"\n{'='*60}")
        print("EVALUATION ONLY MODE")
        print(f"{'='*60}")
        print(f"Loading checkpoint: {args.eval_only}")

        checkpoint = torch.load(args.eval_only, weights_only=False, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.best_val_r2 = checkpoint.get('best_val_r2', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Best val R²: {trainer.best_val_r2:.4f}")
        print(f"Running final evaluation on test set...")

        # Run final evaluation (this will save test_predictions.npz)
        results = trainer._final_evaluation()
        return results

    # Normal training mode
    results = trainer.train()

    return results


if __name__ == '__main__':
    main()
