#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Online Plotting Module for RL Training Scripts

This module provides consistent plotting functionality across all fixed_*.py
RL training scripts. It uses the SAMBRL-style plotting pattern with:
- 3-column layout for better organization
- Comparison with historical data (model-free baseline + previous runs)
- 95% confidence interval shading
- Best value reference lines
- Synchronized X-axis across all plots

Usage:
    from live_plotter import OnlinePlotter; from server_combined import DATA_KEYS

    # Initialize at the start of training
    plotter = OnlinePlotter(
        data_keys=DATA_KEYS,
        experiment_time=100,
        module_name="SAMBRL",
        save_dir="save_model",
        server_name_1="baseline.csv",  # Model-free baseline
        server_name_2="previous.csv"   # Previous run of same module
    )

    # Update each iteration
    plotter.update(current_run_data_map, iteration)

    # Save final plot
    plotter.save("output_plot.png")
"""

import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import DATA_KEYS from server_combined.py (single source of truth)
from server_combined import DATA_KEYS


def parse_hyperparams_from_filename(filename):
    """
    Parse hyperparameters from a filename.
    Returns dict with keys: exp, beta, lr, eps_min, batch, module_name
    Returns None if parsing fails.

    Supports formats:
    - MODULE_YYYYMMDD_HHMMSS_EXPep_betaB_lrL_epsE_batchB.csv (new format)
    - MODULE_YYYYMMDD_HHMMSS_EXPep_betaB.csv (old format)
    """
    basename = os.path.basename(filename)

    # Try new format: MODULE_YYYYMMDD_HHMMSS_EXPep_betaB_lrL_epsE_batchB.csv
    # Use (.+?) non-greedy to handle module names with underscores like SAMBRL_D3QN
    pattern_new = r'^(.+?)_(\d{8}_\d{6})_(\d+)ep_beta([\d.]+)_lr([\d.]+)_eps([\d.]+)_batch(\d+)\.csv$'
    match = re.match(pattern_new, basename)
    if match:
        return {
            'module_name': match.group(1),
            'timestamp': match.group(2),
            'exp': int(match.group(3)),
            'beta': float(match.group(4)),
            'lr': float(match.group(5)),
            'eps_min': float(match.group(6)),
            'batch': int(match.group(7))
        }

    # Try old format: MODULE_YYYYMMDD_HHMMSS_EXPep_betaB.csv
    pattern_old = r'^(.+?)_(\d{8}_\d{6})_(\d+)ep_beta([\d.]+)\.csv$'
    match = re.match(pattern_old, basename)
    if match:
        return {
            'module_name': match.group(1),
            'timestamp': match.group(2),
            'exp': int(match.group(3)),
            'beta': float(match.group(4)),
            'lr': None,
            'eps_min': None,
            'batch': None
        }

    return None


def load_historical_data(filepath, data_keys, target_rows=None):
    """Load historical training data from CSV for comparison plotting.

    Args:
        filepath: Path to the CSV file
        data_keys: List of column names to load
        target_rows: If specified and data has more rows, aggregate to this count
                    (for cross-module comparison where different modules have different row counts)

    Returns:
        Dictionary mapping data_keys to lists of values
    """
    data = {key: [] for key in data_keys}
    if not filepath or not os.path.exists(filepath):
        logging.debug(f"No historical data found at {filepath}")
        return data

    logging.info(f"Loading historical data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        actual_rows = len(df)

        # Check if we need to aggregate (cross-module comparison)
        if target_rows and actual_rows != target_rows:
            if actual_rows > target_rows and actual_rows % target_rows == 0:
                # Aggregate by taking mean of each group
                group_size = actual_rows // target_rows
                logging.info(f"Aggregating {actual_rows} rows to {target_rows} (group_size={group_size})")

                for key in data_keys:
                    if key in df.columns:
                        values = df[key].tolist()
                        aggregated = []
                        for i in range(0, len(values), group_size):
                            group = values[i:i + group_size]
                            aggregated.append(np.mean(group))
                        data[key] = aggregated
            elif actual_rows < target_rows:
                # Fewer rows - use directly
                logging.info(f"Using {actual_rows} rows directly (fewer than target {target_rows})")
                for key in data_keys:
                    if key in df.columns:
                        data[key] = df[key].tolist()
            else:
                logging.warning(f"Cannot aggregate {actual_rows} rows to {target_rows} (not evenly divisible)")
                for key in data_keys:
                    if key in df.columns:
                        data[key] = df[key].tolist()
        else:
            # No aggregation needed - load directly
            for key in data_keys:
                if key in df.columns:
                    data[key] = df[key].tolist()

    except Exception as e:
        logging.warning(f"Failed to load historical data: {e}")
    return data


def extract_server_name(tuning_name):
    """Extract server algorithm name from filename for plot legend.

    Returns descriptive names like 'SAMBRL (Model-Based)' or 'SAMFRL (Model-Free)'
    """
    if not tuning_name:
        return "Unknown"

    basename = os.path.basename(tuning_name)
    upper_basename = basename.upper()

    # Multi-agent modules
    if "MAMBRL" in upper_basename:
        return "MAMBRL (Model-Based)"
    elif "MAMFRL" in upper_basename:
        return "MAMFRL (Model-Free)"
    elif "MARBRL" in upper_basename:
        return "MARBRL (Reward-Based)"
    elif "MARB_D3QN" in upper_basename or "MARB-D3QN" in upper_basename:
        return "MARB_D3QN (Reward-Based)"
    elif "MARL_GEAR" in upper_basename:
        return "MARL_GEAR"
    elif "MAML" in upper_basename:
        return "MAML (Meta-Learning)"

    # Single-agent modules
    if "SAMBRL" in upper_basename:
        return "SAMBRL (Model-Based)"
    elif "SAMFRL" in upper_basename:
        return "SAMFRL (Model-Free)"
    elif "SARBRL" in upper_basename:
        return "SARBRL (Reward-Based)"
    elif "SAMBPGRL" in upper_basename:
        return "SAMBPGRL (Policy Gradient)"

    # Fallback: try old pattern
    match = re.search(r"server_(.*?)_\d+", tuning_name)
    if match:
        return match.group(1)

    # Last resort: use filename prefix
    if '_' in basename:
        return basename.split('_')[0]

    return "Historical"


class OnlinePlotter:
    """
    Online plotter for RL training with SAMBRL-style visualization.

    Features:
    - 3-column subplot layout
    - Historical data comparison (baseline + previous runs)
    - 95% confidence interval shading
    - Best value reference lines
    - Synchronized X-axis
    """

    def __init__(self, data_keys=None, experiment_time=100, module_name="RL",
                 save_dir="save_model", server_name_1='', server_name_2='',
                 baseline_label=None, previous_label=None):
        """
        Initialize the online plotter.

        Args:
            data_keys: List of metrics to plot (default: DATA_KEYS)
            experiment_time: Total number of iterations expected
            module_name: Name of the current RL module (e.g., "SAMBRL")
            save_dir: Directory containing historical data and for saving plots
            server_name_1: Filename for baseline comparison (e.g., model-free version)
            server_name_2: Filename for previous run comparison
            baseline_label: Custom label for baseline (auto-detected if None)
            previous_label: Custom label for previous run (auto-detected if None)
        """
        self.data_keys = data_keys if data_keys is not None else DATA_KEYS
        self.experiment_time = experiment_time
        self.module_name = module_name
        self.save_dir = save_dir
        self.ts = []

        # Determine labels
        self.baseline_label = baseline_label or extract_server_name(server_name_1)
        self.previous_label = previous_label or extract_server_name(server_name_2)
        self.current_label = f"{module_name} (Current)"

        # Setup figure with 3 columns
        num_plots = len(self.data_keys)
        ncols = 3
        nrows = (num_plots + ncols - 1) // ncols
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        self.axs = self.axs.flatten()
        self.fig.tight_layout(pad=4.0)

        # Load historical data for comparison
        self.hist_data_1 = load_historical_data(
            os.path.join(save_dir, server_name_1) if server_name_1 else "",
            self.data_keys,
            target_rows=experiment_time
        )
        self.hist_data_2 = load_historical_data(
            os.path.join(save_dir, server_name_2) if server_name_2 else "",
            self.data_keys,
            target_rows=experiment_time
        )

        # Track best values for reference lines
        self.best_values = {}

    def update(self, current_run_data_map, iteration):
        """
        Update plots with current iteration data.

        Args:
            current_run_data_map: Dictionary mapping data_keys to lists of values
            iteration: Current iteration number
        """
        self.ts.append(iteration)

        for i, key in enumerate(self.data_keys):
            if i >= len(self.axs):
                break

            ax = self.axs[i]
            ax.clear()

            h_data_1 = self.hist_data_1.get(key, [])
            h_data_2 = self.hist_data_2.get(key, [])
            current_data = current_run_data_map.get(key, [])

            # Plot baseline comparison (model-free - red dashed)
            if h_data_1:
                ax.plot(range(len(h_data_1)), h_data_1,
                        label=self.baseline_label,
                        linestyle='--', color='red', linewidth=2)

            # Plot historical same-module data (green dashed)
            if h_data_2:
                ax.plot(range(len(h_data_2)), h_data_2,
                        label=self.previous_label,
                        linestyle=':', color='green', linewidth=1.5, alpha=0.7)

            # Plot current run with 95% CI
            length = min(len(self.ts), len(current_data))
            if length > 0:
                ax.plot(self.ts[:length], current_data[:length],
                        label=self.current_label, color='blue', linewidth=2.5)

                # Add 95% confidence interval shading
                if length > 10:
                    window_size = min(10, length // 5)
                    if window_size >= 2:
                        series = pd.Series(current_data[:length])
                        rolling = series.rolling(window=window_size, center=True, min_periods=1)
                        rolling_mean = rolling.mean()
                        rolling_std = rolling.std()
                        upper_bound = rolling_mean + 1.96 * rolling_std
                        lower_bound = rolling_mean - 1.96 * rolling_std
                        ax.fill_between(self.ts[:length], lower_bound, upper_bound,
                                        alpha=0.2, color='blue', label='95% CI')

            # Add best value reference lines for key metrics
            if key == 'makespan' and current_data:
                best_val = min(current_data)
                self.best_values['makespan'] = best_val
                ax.axhline(y=best_val, color='purple', linestyle=':',
                           linewidth=1.5, label=f'Best: {best_val:.2f}s')
            elif key == 'reward' and current_data:
                best_val = max(current_data)
                self.best_values['reward'] = best_val
                ax.axhline(y=best_val, color='purple', linestyle=':',
                           linewidth=1.5, label=f'Best: {best_val:.2f}')

            # Synchronize X-axis
            max_x = max(
                len(h_data_1) if h_data_1 else 0,
                len(h_data_2) if h_data_2 else 0,
                length,
                self.experiment_time
            )
            ax.set_xlim(0, max_x)

            # Labels and formatting
            ax.set_ylabel(key, fontsize=10, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_title(f'{key.upper()}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=7)

        plt.tight_layout()
        plt.pause(0.1)

    def save(self, filename):
        """
        Save the current plot to a file.

        Args:
            filename: Output filename (can be full path or just filename)
        """
        if os.path.dirname(filename):
            output_path = filename
        else:
            output_path = os.path.join(self.save_dir, filename)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else self.save_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot to '{output_path}'")

    def close(self):
        """Close the plot figure."""
        plt.close(self.fig)


def create_data_map(total_makespan_data, num_cores_data, cores_combination_data,
                    profiler_qmax_data, total_energy_data, freq_data,
                    priority_data, thermal_data, total_branch_misses_data,
                    total_cache_misses_data, priority_values_data, total_rewards,
                    thermal_qmax_data=None, priority_qmax_data=None):
    """
    Create data map for plotting from common tracking lists.

    For multi-agent modules (MAMBRL, MAMFRL, etc.), pass the additional
    thermal and priority Q-max data to compute combined values.

    Args:
        total_makespan_data: List of makespan values per iteration
        num_cores_data: List of core counts per iteration
        cores_combination_data: List of core bitmasks per iteration
        profiler_qmax_data: List of profiler Q-max values
        total_energy_data: List of energy consumption values per iteration
        freq_data: List of frequency steps per iteration
        priority_data: List of priority tuple integers
        thermal_data: List of temperature values
        total_branch_misses_data: List of branch misses
        total_cache_misses_data: List of cache misses
        priority_values_data: List of mean priority values
        total_rewards: List of total rewards
        thermal_qmax_data: Optional list of thermal Q-max (multi-agent only)
        priority_qmax_data: Optional list of priority Q-max (multi-agent only)

    Returns:
        Dictionary mapping DATA_KEYS to data lists
    """
    # Compute combined Q-max for multi-agent modules
    if thermal_qmax_data is not None and priority_qmax_data is not None:
        qmax_data = [
            max(pq, tq, prq) for pq, tq, prq in
            zip(profiler_qmax_data, thermal_qmax_data, priority_qmax_data)
        ]
    else:
        qmax_data = profiler_qmax_data

    return {
        'makespan': total_makespan_data,
        'num_cores': num_cores_data,
        'cores': cores_combination_data,
        'qmax': qmax_data,
        'energy': total_energy_data,
        'freq': freq_data,
        'priority_combination': priority_data,
        'thermal': thermal_data,
        'branchmisses': total_branch_misses_data,
        'cachemisses': total_cache_misses_data,
        'priority': priority_values_data,
        'reward': total_rewards,
    }


if __name__ == "__main__":
    # Example usage / test
    logging.basicConfig(level=logging.INFO)

    print("live_plotter.py - Unified Online Plotting Module")
    print("=" * 50)
    print("This module should be imported by fixed_*.py training scripts.")
    print()
    print("Available exports:")
    print("  - DATA_KEYS: Standard list of metrics to plot")
    print("  - OnlinePlotter: Class for real-time plotting during training")
    print("  - create_data_map: Helper to create data dict from tracking lists")
    print("  - load_historical_data: Load CSV for comparison")
    print("  - extract_server_name: Get legend label from filename")
    print("  - parse_hyperparams_from_filename: Parse hyperparams from CSV name")
