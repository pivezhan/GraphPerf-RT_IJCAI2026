#!/usr/bin/env python3
"""
Generate hyperparameter configurations for parallel SLURM job array.

This script creates a JSON file containing all hyperparameter combinations,
each assigned a unique task ID for use with SLURM job arrays.
"""

import json
import itertools
from pathlib import Path
from datetime import datetime


def generate_configs():
    """Generate all hyperparameter combinations."""

    # Hyperparameter grid
    param_grid = {
        'lambda_mse': [1.0, 5.0, 10.0, 20.0],
        'lambda_ns': [0.001, 0.01, 0.1],
        'lr': [0.0005, 0.001],
        'hidden_dim': [128, 256]
    }

    # Fixed parameters
    fixed_params = {
        'epochs': 50,  # Shorter for search
        'batch_size': 64,
        'device': 'cpu'  # Due to torch_scatter CUDA compatibility
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = []
    for i, combo in enumerate(itertools.product(*values)):
        config = {
            'task_id': i,
            'experiment_name': f"hp_{i:03d}_lmse{combo[0]}_lns{combo[1]}_lr{combo[2]}_hd{combo[3]}",
            **dict(zip(keys, combo)),
            **fixed_params
        }
        configs.append(config)

    return configs, param_grid


def main():
    # Generate configurations
    configs, param_grid = generate_configs()

    # Create output directory
    output_dir = Path(__file__).parent / 'hyperparam_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configurations
    search_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = output_dir / f'configs_{search_id}.json'

    output = {
        'search_id': search_id,
        'total_configs': len(configs),
        'param_grid': param_grid,
        'configs': configs
    }

    with open(config_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Also save as latest for easy reference
    latest_file = output_dir / 'configs_latest.json'
    with open(latest_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print("Hyperparameter Search Configuration Generator")
    print("=" * 60)
    print(f"\nSearch ID: {search_id}")
    print(f"Total configurations: {len(configs)}")
    print(f"\nParameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print(f"\nConfigurations saved to:")
    print(f"  {config_file}")
    print(f"  {latest_file}")
    print(f"\nTo run parallel search:")
    print(f"  sbatch --array=0-{len(configs)-1} slurm_array.sh")
    print(f"\nOr run subset (e.g., first 16):")
    print(f"  sbatch --array=0-15 slurm_array.sh")
    print("=" * 60)

    return config_file


if __name__ == '__main__':
    main()
