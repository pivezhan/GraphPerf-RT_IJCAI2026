#!/usr/bin/env python3
"""Evaluate trained model and compute real-world metrics."""

import argparse
import torch
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.gat import create_model


@torch.no_grad()
def evaluate(model, loader, device, scaler_params, use_log=False):
    """Evaluate model and denormalize predictions."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.global_features, batch.batch)
        
        all_preds.append(out.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())
    
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Denormalize predictions
    if use_log:
        # Step 1: Reverse z-score normalization on log values
        target_idx = scaler_params['columns'].index('time_elapsed_log')
        mean = scaler_params['mean'][target_idx]
        std = scaler_params['std'][target_idx]
        
        preds_log = preds * std + mean
        targets_log = targets * std + mean
        
        # Step 2: Reverse log transform to get actual time
        preds_real = np.expm1(preds_log)
        targets_real = np.expm1(targets_log)
    else:
        target_idx = scaler_params['columns'].index('time_elapsed')
        mean = scaler_params['mean'][target_idx]
        std = scaler_params['std'][target_idx]
        
        preds_real = preds * std + mean
        targets_real = targets * std + mean
    
    # Compute metrics
    mse = np.mean((preds_real - targets_real) ** 2)
    mae = np.mean(np.abs(preds_real - targets_real))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((preds_real - targets_real) / np.maximum(targets_real, 0.01))) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'preds': preds_real,
        'targets': targets_real
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='best.pth')
    args = parser.parse_args()
    
    exp_dir = Path('experiments/checkpoints') / args.experiment
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(exp_dir / args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    model_type = config['model'].get('type', 'gat')
    target_col = config.get('data', {}).get('target_column', 'time_elapsed')
    use_log = 'log' in target_col
    
    print(f"Model type: {model_type}")
    print(f"Target column: {target_col}")
    print(f"Using log transform: {use_log}")
    
    # FIXED: Load metadata correctly
    metadata_path = config['paths'].get('metadata', 'data/processed/merged_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Handle different metadata formats
    if 'scaler_params' in metadata:
        scaler_params = metadata['scaler_params']
    else:
        scaler_params = metadata
    
    print("\nLoading test data...")
    use_cfg = config.get('data', {}).get('use_cfg_features', False)
    _, _, test_loader, _ = create_dataloaders(
        train_path=config['paths']['train'],
        val_path=config['paths']['val'],
        test_path=config['paths']['test'],
        batch_size=config['training']['batch_size'],
        num_workers=0,
        metadata_path=metadata_path,
        target_column=target_col,
        use_cfg_features=use_cfg
    )
    
    print("Loading model...")
    if model_type == 'mlp':
        from src.models.baseline import create_mlp_model
        model = create_mlp_model(config['model']).to(device)
    else:
        model = create_model(config['model']).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Evaluating...")
    results = evaluate(model, test_loader, device, scaler_params, use_log=use_log)
    
    print(f"\n{'='*60}")
    print(f"Test Set Results (Real Scale - Time in Seconds)")
    print(f"{'='*60}")
    print(f"  MSE:  {results['mse']:.4f} s²")
    print(f"  RMSE: {results['rmse']:.4f} s")
    print(f"  MAE:  {results['mae']:.4f} s")
    print(f"  MAPE: {results['mape']:.2f}%")
    
    print(f"\nSample Predictions (first 10):")
    print(f"  {'Predicted':>12s}  {'Actual':>12s}  {'Error':>12s}  {'Error %':>10s}")
    for i in range(min(10, len(results['preds']))):
        pred = results['preds'][i]
        target = results['targets'][i]
        error = pred - target
        error_pct = (error / max(target, 0.01) * 100)
        print(f"  {pred:12.4f}  {target:12.4f}  {error:12.4f}  {error_pct:10.1f}%")
    
    save_dict = {k: v for k, v in results.items() if k not in ['preds', 'targets']}
    with open(exp_dir / 'test_results.json', 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {exp_dir}/test_results.json")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
