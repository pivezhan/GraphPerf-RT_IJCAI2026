#!/usr/bin/env python3
"""
Aggregate results from parallel hyperparameter search.

This script collects results from all completed SLURM array tasks,
finds the best configuration, and generates a summary report.

Usage:
    python aggregate_sweep.py                    # Use latest search
    python aggregate_sweep.py --search_id XXX   # Use specific search
    python aggregate_sweep.py --run_best        # Run best config with more epochs
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_results(results_dir: Path) -> list:
    """Load all result files from the results directory."""
    results = []

    for result_file in results_dir.glob('task_*.json'):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")

    return results


def create_summary_dataframe(results: list) -> pd.DataFrame:
    """Create a pandas DataFrame from results."""
    rows = []

    for r in results:
        row = {
            'task_id': r.get('task_id', -1),
            'experiment': r.get('experiment_name', 'unknown'),
            'lambda_mse': r.get('lambda_mse', -1),
            'lambda_ns': r.get('lambda_ns', -1),
            'lr': r.get('lr', -1),
            'hidden_dim': r.get('hidden_dim', -1),
            'test_r2': r.get('test_r2', -999),
            'test_rmse': r.get('test_rmse', 999),
            'test_mae': r.get('test_mae', 999),
            'test_mape': r.get('test_mape', 999),
            'test_spearman': r.get('test_spearman', -999),
            'test_ece': r.get('test_ece', 999),
            'test_mce': r.get('test_mce', 999),
            'test_picp_95': r.get('test_picp_95', -999),
            'test_mpiw': r.get('test_mpiw', 999),
            'status': 'completed' if r.get('test_r2') is not None else r.get('status', 'unknown')
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('test_r2', ascending=False)

    return df


def print_summary(df: pd.DataFrame, top_n: int = 10):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("=" * 80)

    completed = df[df['status'] == 'completed']
    failed = df[df['status'] != 'completed']

    print(f"\nTotal configurations: {len(df)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    if len(completed) == 0:
        print("\nNo completed results found!")
        return None

    print(f"\n{'=' * 80}")
    print(f"TOP {min(top_n, len(completed))} CONFIGURATIONS (by Test R²)")
    print("=" * 80)

    display_cols = ['task_id', 'lambda_mse', 'lambda_ns', 'lr', 'hidden_dim',
                    'test_r2', 'test_rmse', 'test_ece', 'test_picp_95']

    print(completed[display_cols].head(top_n).to_string(index=False))

    # Best configuration
    best = completed.iloc[0]

    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"\nTask ID: {best['task_id']}")
    print(f"Experiment: {best['experiment']}")
    print(f"\nHyperparameters:")
    print(f"  lambda_mse: {best['lambda_mse']}")
    print(f"  lambda_ns:  {best['lambda_ns']}")
    print(f"  lr:         {best['lr']}")
    print(f"  hidden_dim: {best['hidden_dim']}")
    print(f"\nMetrics (for IJCAI Paper):")
    print(f"  Test R²:       {best['test_r2']:.4f}")
    print(f"  Test RMSE:     {best['test_rmse']:.4f}")
    print(f"  Test MAE:      {best['test_mae']:.4f}")
    print(f"  Test MAPE:     {best['test_mape']:.2f}%")
    print(f"  Test Spearman: {best['test_spearman']:.4f}")
    print(f"  Test ECE:      {best['test_ece']:.4f}")
    print(f"  Test MCE:      {best['test_mce']:.4f}")
    print(f"  Test PICP@95%: {best['test_picp_95']:.4f}")
    print(f"  Test MPIW:     {best['test_mpiw']:.4f}")

    return best


def generate_best_config_command(best: pd.Series, epochs: int = 100) -> str:
    """Generate command to run best config with more epochs."""
    cmd = f"""
# Run best configuration with {epochs} epochs
python train.py \\
    --experiment "nig_best_$(date +%Y%m%d_%H%M%S)" \\
    --epochs {epochs} \\
    --batch_size 64 \\
    --lr {best['lr']} \\
    --hidden_dim {int(best['hidden_dim'])} \\
    --lambda_mse {best['lambda_mse']} \\
    --lambda_ns {best['lambda_ns']} \\
    --device cpu
"""
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Aggregate hyperparameter search results')
    parser.add_argument('--search_id', type=str, default=None,
                       help='Search ID (default: use latest)')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Results directory path')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top results to show')
    parser.add_argument('--run_best', action='store_true',
                       help='Print command to run best config')
    parser.add_argument('--export_csv', type=str, default=None,
                       help='Export results to CSV file')

    args = parser.parse_args()

    # Determine results directory
    base_dir = Path(__file__).parent / 'hyperparam_search'

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = base_dir / 'results'

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run hyperparameter search first.")
        return

    # Load results
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)

    if not results:
        print("No results found!")
        return

    # Create summary
    df = create_summary_dataframe(results)

    # Print summary
    best = print_summary(df, args.top_n)

    # Export to CSV if requested
    if args.export_csv:
        csv_path = Path(args.export_csv)
        df.to_csv(csv_path, index=False)
        print(f"\nResults exported to: {csv_path}")
    else:
        # Save to default location
        csv_path = base_dir / 'search_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

    # Print command to run best config
    if best is not None and args.run_best:
        print("\n" + "=" * 80)
        print("COMMAND TO RUN BEST CONFIGURATION")
        print("=" * 80)
        print(generate_best_config_command(best))

    # Save best config for reference
    if best is not None:
        best_config = {
            'task_id': int(best['task_id']),
            'lambda_mse': float(best['lambda_mse']),
            'lambda_ns': float(best['lambda_ns']),
            'lr': float(best['lr']),
            'hidden_dim': int(best['hidden_dim']),
            'test_r2': float(best['test_r2']),
            'test_rmse': float(best['test_rmse']),
            'test_ece': float(best['test_ece']),
            'test_picp_95': float(best['test_picp_95']),
            'timestamp': datetime.now().isoformat()
        }

        best_config_path = base_dir / 'best_config.json'
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest config saved to: {best_config_path}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
