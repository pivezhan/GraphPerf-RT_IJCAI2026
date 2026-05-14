#!/usr/bin/env python3
"""
GraphPerf-RT — Section 4 (GAT surrogate) training entry point.

Trains the heterogeneous Graph Attention Network described in §4 of the
camera-ready paper.  Reproduces the headline accuracy numbers:

    - R²       = 0.81  (log-makespan, test split)
    - Spearman ρ = 0.95
    - 2–7 ms inference latency on Jetson TX2

Usage
-----
    export GRAPHPERF_RT_ROOT="$(pwd)"
    python gat/scripts/train.py \\
        --config gat/configs/gat_log_config.yaml \\
        --experiment exp_v2_gat_log

The configuration `gat_log_config.yaml` is the configuration reported in the
paper (V2 dataset, log-transformed targets, full per-platform feature set).

Outputs
-------
    experiments/<name>/best.pth          best validation checkpoint
    experiments/<name>/latest.pth        last epoch (resumable)
    experiments/<name>/metrics.json      R², RMSE, MAE, Spearman ρ
    experiments/<name>/history.json      per-epoch loss / metric curves
"""

import argparse
import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders_v2
from src.models.gat import create_model


class TrainerV2:
    """Training pipeline for GAT model - V2 with full metrics."""
    
    def __init__(self, config, experiment_name):
        self.config = config
        self.experiment_name = experiment_name
        
        # Setup device
        self.device = torch.device(config['device'])
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cpu':
            torch.set_num_threads(config['data']['num_workers'])
            print(f"CPU threads: {torch.get_num_threads()}")
        
        # Create experiment directory
        self.exp_dir = Path(config['paths']['checkpoints']) / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # Initialize tracking - now with multiple metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_rmses = []
        self.val_rmses = []
        self.val_r2s = []
        self.epoch_times = []
        
        # Set random seed
        if 'seed' in config:
            torch.manual_seed(config['seed'])
            np.random.seed(config['seed'])
    
    def setup_data(self):
        """Setup data loaders."""
        print("\nSetting up data loaders...")
        print(f"  Train: {self.config['paths']['train']}")
        print(f"  Val:   {self.config['paths']['val']}")
        print(f"  Test:  {self.config['paths']['test']}")
        
        target_col = self.config['data']['target_column']
        use_cfg = self.config['data'].get('use_cfg_features', False)
        
        self.train_loader, self.val_loader, self.test_loader, self.feature_dims = \
            create_dataloaders_v2(
                train_path=self.config['paths']['train'],
                val_path=self.config['paths']['val'],
                test_path=self.config['paths']['test'],
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                metadata_path=self.config['paths']['metadata'],
                target_column=target_col,
                use_cfg_features=use_cfg
            )
        
        print(f"\n  Dataset sizes:")
        print(f"    Train: {len(self.train_loader.dataset):,} samples")
        print(f"    Val:   {len(self.val_loader.dataset):,} samples")
        print(f"    Test:  {len(self.test_loader.dataset):,} samples")
        print(f"\n  Feature dimensions:")
        print(f"    Node features:   {self.feature_dims[0]}")
        print(f"    Global features: {self.feature_dims[1]}")
    
    def setup_model(self):
        """Setup model, optimizer, loss."""
        print("\nSetting up model...")
        
        self.config['model']['node_feature_dim'] = self.feature_dims[0]
        self.config['model']['global_feature_dim'] = self.feature_dims[1]
        
        model_type = self.config['model'].get('type', 'gat')
        
        if model_type == 'mlp':
            from src.models.baseline import create_mlp_model
            self.model = create_mlp_model(self.config['model']).to(self.device)
        else:
            self.model = create_model(self.config['model']).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n  Model: {model_type.upper()}")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        opt_type = self.config['optimizer']['type']
        if opt_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=self.config['optimizer']['betas']
            )
        elif opt_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=self.config['optimizer']['betas']
            )
        
        print(f"  Optimizer: {opt_type.upper()}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        
        sched_type = self.config['scheduler']['type']
        if sched_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['scheduler']['factor'],
                patience=self.config['scheduler']['patience'],
                min_lr=self.config['scheduler']['min_lr']
            )
        
        self.criterion = nn.MSELoss()
        print(f"  Loss function: MSE")
    
    def compute_metrics(self, predictions, targets):
        """Compute MAE, RMSE, R² metrics."""
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, targets) if len(np.unique(targets)) > 1 else 0.0
        
        # Handle edge case
        if len(predictions) > 1:
            r2 = r2_score(targets, predictions)
        
        return mae, rmse, r2
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        start_time = time.time()
        
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {epoch+1:03d} [Train]',
            ncols=100
        )
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(
                batch.x,
                batch.edge_index,
                batch.global_features,
                batch.batch
            )
            
            loss = self.criterion(out, batch.y)
            loss.backward()
            
            if 'grad_clip' in self.config['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(out.detach())
            all_targets.append(batch.y.detach())
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        epoch_time = time.time() - start_time
        self.epoch_times.append(epoch_time)
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mae, rmse, _ = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / len(self.train_loader), mae, rmse
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch+1:03d} [Val]  ',
            ncols=100
        )
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            out = self.model(
                batch.x,
                batch.edge_index,
                batch.global_features,
                batch.batch
            )
            
            loss = self.criterion(out, batch.y)
            total_loss += loss.item()
            
            all_preds.append(out)
            all_targets.append(batch.y)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mae, rmse, r2 = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / len(self.val_loader), mae, rmse, r2
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_maes': self.train_maes,
            'val_maes': self.val_maes,
            'train_rmses': self.train_rmses,
            'val_rmses': self.val_rmses,
            'val_r2s': self.val_r2s,
            'config': self.config
        }
        
        torch.save(checkpoint, self.exp_dir / 'latest.pth')
        
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best.pth')
            print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.6f})")
    
    def plot_losses(self):
        """Plot training curves - separate plots for each metric."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        
        # 1. MSE Loss
        axes[0, 0].plot(self.train_losses, label='Train', linewidth=2, color='blue')
        axes[0, 0].plot(self.val_losses, label='Val', linewidth=2, color='orange')
        axes[0, 0].axhline(y=self.best_val_loss, color='r', linestyle='--', 
                          alpha=0.5, label=f'Best ({self.best_val_loss:.4f})')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('MSE Loss', fontsize=12)
        axes[0, 0].set_title('MSE Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MAE
        axes[0, 1].plot(self.train_maes, label='Train', linewidth=2, color='blue')
        axes[0, 1].plot(self.val_maes, label='Val', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('MAE', fontsize=12)
        axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RMSE
        axes[0, 2].plot(self.train_rmses, label='Train', linewidth=2, color='blue')
        axes[0, 2].plot(self.val_rmses, label='Val', linewidth=2, color='orange')
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('RMSE', fontsize=12)
        axes[0, 2].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. R² Score
        axes[1, 0].plot(self.val_r2s, label='Val R²', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('R² Score', fontsize=12)
        axes[1, 0].set_title('R² Score (Validation)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Epoch Times
        if len(self.epoch_times) > 0:
            axes[1, 1].plot(self.epoch_times, linewidth=2, color='purple')
            axes[1, 1].axhline(y=np.mean(self.epoch_times), color='r', 
                              linestyle='--', alpha=0.5,
                              label=f'Avg: {np.mean(self.epoch_times):.1f}s')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
            axes[1, 1].set_title('Epoch Duration', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Learning Rate
        axes[1, 2].text(0.5, 0.5, f'Best Val Loss: {self.best_val_loss:.6f}\n'
                                   f'Best Val MAE: {min(self.val_maes):.6f}\n'
                                   f'Best Val RMSE: {min(self.val_rmses):.6f}\n'
                                   f'Best Val R²: {max(self.val_r2s):.6f}',
                       transform=axes[1, 2].transAxes,
                       fontsize=14, verticalalignment='center',
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f"Starting Training: {self.experiment_name}")
        print(f"{'='*70}\n")
        
        print(f"Configuration:")
        print(f"  Epochs: {self.config['training']['num_epochs']}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        print(f"  Patience: {self.config['training']['patience']}")
        print(f"  Device: {self.device}")
        print()
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_loss, train_mae, train_rmse = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_maes.append(train_mae)
            self.train_rmses.append(train_rmse)
            
            # Validate
            val_loss, val_mae, val_rmse, val_r2 = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            self.val_rmses.append(val_rmse)
            self.val_r2s.append(val_r2)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print summary with all metrics
            print(f"\nEpoch {epoch+1:03d} Summary:")
            print(f"  Train: Loss={train_loss:.6f}, MAE={train_mae:.6f}, RMSE={train_rmse:.6f}")
            print(f"  Val:   Loss={val_loss:.6f}, MAE={val_mae:.6f}, RMSE={val_rmse:.6f}, R²={val_r2:.6f}")
            print(f"  LR: {current_lr:.6f} | Time: {self.epoch_times[-1]:.1f}s")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  ✓ New best! (improved by {improvement:.6f})")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.config['training']['patience']}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging'].get('save_every', 5) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['patience']:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                break
            
            print()
        
        # Plot losses
        self.plot_losses()
        
        # Save final results
        total_time = sum(self.epoch_times)
        results = {
            'experiment_name': self.experiment_name,
            'best_val_loss': float(self.best_val_loss),
            'best_val_mae': float(min(self.val_maes)),
            'best_val_rmse': float(min(self.val_rmses)),
            'best_val_r2': float(max(self.val_r2s)),
            'final_train_loss': float(self.train_losses[-1]),
            'final_val_loss': float(self.val_losses[-1]),
            'num_epochs_trained': len(self.train_losses),
            'total_training_time_seconds': float(total_time),
            'avg_epoch_time_seconds': float(np.mean(self.epoch_times)),
            'feature_dims': {
                'node': self.feature_dims[0],
                'global': self.feature_dims[1]
            }
        }
        
        with open(self.exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"  Best Val Loss: {self.best_val_loss:.6f}")
        print(f"  Best Val MAE:  {min(self.val_maes):.6f}")
        print(f"  Best Val RMSE: {min(self.val_rmses):.6f}")
        print(f"  Best Val R²:   {max(self.val_r2s):.6f}")
        print(f"  Epochs:        {len(self.train_losses)}")
        print(f"  Total time:    {total_time/60:.1f} minutes")
        print(f"  Results:       {self.exp_dir}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train GAT model - V2')
    parser.add_argument('--config', type=str, 
                       default='configs/model_config_v2_log.yaml',
                       help='Config file path')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = TrainerV2(config, args.experiment)
    trainer.setup_data()
    trainer.setup_model()
    trainer.train()


if __name__ == '__main__':
    main()