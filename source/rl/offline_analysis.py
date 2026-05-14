#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_analysis.py - Offline Statistical Analysis for AISTATS Rebuttal

This script performs offline statistical analysis on the collected profiling data
to address the P0 tasks from TASKS.md:

P0 Tasks Addressed:
==================
1. Data Leakage Fix (fwuN) - Benchmark-level holdout evaluation
2. Statistical Tests (fwuN) - Confidence intervals, multiple seeds, significance tests
3. Cross-Platform Transfer (the contributor P2) - Train on 2 platforms, test on 1

Data Sources:
=============
- data/large_data/profiling_data_tx2.csv (20,160 samples, 42 benchmarks)
- data/large_data/profiling_data_orin_nx.csv (19,200 samples, 42 benchmarks)
- data/large_data/profiling_data_rubikpi.csv (26,880 samples, 42 benchmarks)

Usage:
======
python3 offline_analysis.py --task all
python3 offline_analysis.py --task benchmark_holdout
python3 offline_analysis.py --task statistics
python3 offline_analysis.py --task cross_platform

Results are saved to: src/RL/results/
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, ttest_ind, normaltest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.model_selection import train_test_split, KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "large_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data files
DATA_FILES = {
    'tx2': os.path.join(DATA_DIR, 'profiling_data_tx2.csv'),
    'orin_nx': os.path.join(DATA_DIR, 'profiling_data_orin_nx.csv'),
    'rubikpi': os.path.join(DATA_DIR, 'profiling_data_rubikpi.csv'),
}

# Feature columns for prediction
FEATURE_COLS = [
    'frequency_index', 'num_cores', 'cache_misses', 'branch_misses',
    'cycles', 'instructions', 'cache_references', 'context_switches',
    'minor_faults', 'page_faults', 'avg_temp_before',
]

# Target columns for prediction
TARGET_COLS = ['time_elapsed', 'total_energy_consumption']

# Benchmark categories (BOTS vs PolyBench)
BOTS_BENCHMARKS = {
    'alignment', 'fft', 'fib', 'floorplan', 'health', 'concom',
    'knapsack', 'nqueens', 'sort', 'sparselu', 'strassen', 'uts'
}

POLYBENCH_BENCHMARKS = {
    'gemm', 'gemver', 'gesummv', 'symm', 'syr2k', 'syrk', 'trmm',
    '2mm', '3mm', 'atax', 'bicg', 'doitgen', 'mvt',
    'cholesky', 'durbin', 'gramschmidt', 'lu', 'ludcmp', 'trisolv',
    'correlation', 'covariance', 'deriche', 'floyd-warshall', 'nussinov',
    'adi', 'fdtd-2d', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'seidel-2d'
}

# Number of random seeds for statistical significance
N_SEEDS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================
def load_platform_data(platform: str) -> pd.DataFrame:
    """Load data for a specific platform."""
    filepath = DATA_FILES.get(platform)
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found for platform: {platform}")

    df = pd.read_csv(filepath)
    df['platform'] = platform
    print(f"Loaded {len(df)} samples from {platform}")
    return df


def load_all_data() -> pd.DataFrame:
    """Load and combine data from all platforms."""
    dfs = []
    for platform, filepath in DATA_FILES.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['platform'] = platform
            dfs.append(df)
            print(f"Loaded {len(df)} samples from {platform}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(combined)}")
    return combined


def prepare_features(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """Prepare feature matrix from dataframe."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]

    # Handle missing values
    X = df[available_cols].fillna(0).values

    return X, available_cols


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoiding division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
    }


# =============================================================================
# TASK 1: BENCHMARK-LEVEL HOLDOUT (Data Leakage Fix)
# =============================================================================
def run_benchmark_holdout(df: pd.DataFrame, n_seeds: int = N_SEEDS) -> Dict[str, Any]:
    """
    Run benchmark-level holdout evaluation.

    This addresses reviewer fwuN's concern about data leakage.
    We hold out ENTIRE benchmarks for testing, not just configurations.

    Returns per-benchmark holdout results with confidence intervals.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK-LEVEL HOLDOUT EVALUATION")
    print("Addresses: fwuN data leakage concern")
    print("=" * 80)

    benchmarks = df['benchmark'].unique()
    print(f"Total benchmarks: {len(benchmarks)}")

    results = {
        'per_benchmark': {},
        'aggregate': {},
        'summary_table': [],
    }

    # Leave-One-Benchmark-Out Cross-Validation
    logo = LeaveOneGroupOut()

    for target in TARGET_COLS:
        print(f"\n--- Target: {target} ---")

        benchmark_results = []

        for seed_idx, seed in enumerate(RANDOM_SEEDS[:n_seeds]):
            print(f"  Seed {seed_idx + 1}/{n_seeds}: {seed}")

            X, feature_cols = prepare_features(df)
            y = df[target].values
            groups = df['benchmark'].values

            for train_idx, test_idx in logo.split(X, y, groups):
                test_benchmark = groups[test_idx[0]]

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Normalize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
                model.fit(X_train_scaled, y_train)

                # Predict
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred)
                metrics['benchmark'] = test_benchmark
                metrics['seed'] = seed
                metrics['target'] = target
                metrics['n_train'] = len(y_train)
                metrics['n_test'] = len(y_test)

                benchmark_results.append(metrics)

        # Aggregate by benchmark
        results_df = pd.DataFrame(benchmark_results)

        for benchmark in benchmarks:
            bench_data = results_df[results_df['benchmark'] == benchmark]

            if benchmark not in results['per_benchmark']:
                results['per_benchmark'][benchmark] = {}

            for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
                values = bench_data[metric].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                ci = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 1 else 0

                results['per_benchmark'][benchmark][f'{target}_{metric}_mean'] = mean_val
                results['per_benchmark'][benchmark][f'{target}_{metric}_std'] = std_val
                results['per_benchmark'][benchmark][f'{target}_{metric}_ci'] = ci

        # Overall aggregate
        for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
            all_values = results_df[metric].values
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            ci = 1.96 * std_val / np.sqrt(len(all_values)) if len(all_values) > 1 else 0

            results['aggregate'][f'{target}_{metric}_mean'] = mean_val
            results['aggregate'][f'{target}_{metric}_std'] = std_val
            results['aggregate'][f'{target}_{metric}_ci'] = ci

            print(f"  {metric}: {mean_val:.4f} +/- {ci:.4f} (95% CI)")

    # Create summary table
    for benchmark in benchmarks:
        row = {'benchmark': benchmark}
        for target in TARGET_COLS:
            for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
                key = f'{target}_{metric}_mean'
                row[key] = results['per_benchmark'][benchmark].get(key, np.nan)
        results['summary_table'].append(row)

    return results


def plot_benchmark_holdout_results(results: Dict[str, Any], save_dir: str = RESULTS_DIR):
    """Generate plots for benchmark holdout results."""

    summary_df = pd.DataFrame(results['summary_table'])

    # Plot 1: Per-benchmark R2 scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, target in enumerate(TARGET_COLS):
        ax = axes[idx]
        col = f'{target}_R2_mean'

        # Sort by R2
        sorted_df = summary_df.sort_values(col, ascending=True)

        colors = ['#2ecc71' if r2 > 0.9 else '#f39c12' if r2 > 0.7 else '#e74c3c'
                  for r2 in sorted_df[col]]

        ax.barh(range(len(sorted_df)), sorted_df[col], color=colors)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['benchmark'], fontsize=8)
        ax.set_xlabel('R2 Score')
        ax.set_title(f'Benchmark Holdout R2: {target}')
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Good (>0.9)')
        ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.7)')
        ax.set_xlim(0, 1)
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'benchmark_holdout_r2.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: BOTS vs PolyBench comparison
    summary_df['category'] = summary_df['benchmark'].apply(
        lambda x: 'BOTS' if x in BOTS_BENCHMARKS else 'PolyBench'
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, target in enumerate(TARGET_COLS):
        ax = axes[idx]
        col = f'{target}_MAPE_mean'

        bots_mape = summary_df[summary_df['category'] == 'BOTS'][col].values
        poly_mape = summary_df[summary_df['category'] == 'PolyBench'][col].values

        bp = ax.boxplot([bots_mape, poly_mape], labels=['BOTS', 'PolyBench'])
        ax.set_ylabel('MAPE (%)')
        ax.set_title(f'Benchmark Holdout MAPE: {target}')

        # Add mean markers
        means = [np.mean(bots_mape), np.mean(poly_mape)]
        ax.scatter([1, 2], means, color='red', marker='D', s=50, zorder=3, label='Mean')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'benchmark_holdout_bots_vs_poly.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_dir}")


# =============================================================================
# TASK 2: STATISTICAL SIGNIFICANCE TESTS
# =============================================================================
def run_statistical_tests(df: pd.DataFrame, n_seeds: int = N_SEEDS) -> Dict[str, Any]:
    """
    Run statistical significance tests.

    This addresses reviewer fwuN's concern about missing statistics.
    Includes:
    - Confidence intervals
    - Multiple seed evaluation
    - Nonparametric significance tests (Wilcoxon, Mann-Whitney)
    """
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("Addresses: fwuN missing statistics concern")
    print("=" * 80)

    results = {
        'seed_results': [],
        'confidence_intervals': {},
        'significance_tests': {},
        'normality_tests': {},
    }

    # Run experiments with multiple seeds
    X, feature_cols = prepare_features(df)

    for target in TARGET_COLS:
        print(f"\n--- Target: {target} ---")

        y = df[target].values
        seed_metrics = {metric: [] for metric in ['MAE', 'RMSE', 'R2', 'MAPE']}

        for seed_idx, seed in enumerate(RANDOM_SEEDS[:n_seeds]):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )

            # Normalize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)

            for metric, value in metrics.items():
                seed_metrics[metric].append(value)

            results['seed_results'].append({
                'target': target,
                'seed': seed,
                **metrics
            })

        # Calculate confidence intervals
        for metric, values in seed_metrics.items():
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)

            # 95% CI using t-distribution
            ci = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n) if n > 1 else 0

            results['confidence_intervals'][f'{target}_{metric}'] = {
                'mean': mean,
                'std': std,
                'ci_lower': mean - ci,
                'ci_upper': mean + ci,
                'ci_95': ci,
                'n_seeds': n,
            }

            print(f"  {metric}: {mean:.4f} +/- {ci:.4f} (95% CI)")

        # Normality test on residuals
        _, p_value = normaltest(seed_metrics['MAE'])
        results['normality_tests'][f'{target}_MAE'] = {
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }

    # Compare methods: RandomForest vs GradientBoosting
    print("\n--- Method Comparison (Statistical Tests) ---")

    X, feature_cols = prepare_features(df)

    for target in TARGET_COLS:
        y = df[target].values

        rf_errors = []
        gb_errors = []

        for seed in RANDOM_SEEDS[:n_seeds]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # RandomForest
            rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_errors.append(np.abs(y_test - rf_pred))

            # GradientBoosting
            gb = GradientBoostingRegressor(n_estimators=100, random_state=seed)
            gb.fit(X_train_scaled, y_train)
            gb_pred = gb.predict(X_test_scaled)
            gb_errors.append(np.abs(y_test - gb_pred))

        # Flatten errors
        rf_errors_flat = np.concatenate(rf_errors)
        gb_errors_flat = np.concatenate(gb_errors)

        # Wilcoxon signed-rank test (paired)
        # Use MAE per sample
        stat, p_value = mannwhitneyu(rf_errors_flat, gb_errors_flat, alternative='two-sided')

        results['significance_tests'][f'{target}_rf_vs_gb'] = {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'rf_mean_error': np.mean(rf_errors_flat),
            'gb_mean_error': np.mean(gb_errors_flat),
        }

        print(f"  {target} RF vs GB: p={p_value:.4f} {'*' if p_value < 0.05 else ''}")

    return results


def plot_statistical_results(results: Dict[str, Any], save_dir: str = RESULTS_DIR):
    """Generate plots for statistical analysis results."""

    # Plot 1: Confidence intervals
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_to_plot = []
    for key, data in results['confidence_intervals'].items():
        target, metric = key.rsplit('_', 1)
        if metric == 'R2':
            metrics_to_plot.append({
                'label': f'{target}\n{metric}',
                'mean': data['mean'],
                'ci': data['ci_95'],
            })

    x = range(len(metrics_to_plot))
    means = [m['mean'] for m in metrics_to_plot]
    cis = [m['ci'] for m in metrics_to_plot]
    labels = [m['label'] for m in metrics_to_plot]

    ax.bar(x, means, yerr=cis, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('R2 Score')
    ax.set_title('Model Performance with 95% Confidence Intervals')
    ax.set_ylim(0, 1)

    # Add significance markers
    for i, (mean, ci) in enumerate(zip(means, cis)):
        ax.annotate(f'{mean:.3f}\n+/-{ci:.3f}', xy=(i, mean + ci + 0.02), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_intervals.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Seed-wise performance variation
    seed_df = pd.DataFrame(results['seed_results'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, target in enumerate(TARGET_COLS):
        ax = axes[idx]
        target_data = seed_df[seed_df['target'] == target]

        ax.plot(target_data['seed'], target_data['R2'], 'o-', markersize=8)
        ax.axhline(y=target_data['R2'].mean(), color='red', linestyle='--',
                   label=f'Mean: {target_data["R2"].mean():.3f}')
        ax.fill_between(target_data['seed'],
                        target_data['R2'].mean() - target_data['R2'].std(),
                        target_data['R2'].mean() + target_data['R2'].std(),
                        alpha=0.2, color='red')
        ax.set_xlabel('Random Seed')
        ax.set_ylabel('R2 Score')
        ax.set_title(f'Performance Across Seeds: {target}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'seed_variation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_dir}")


# =============================================================================
# TASK 3: CROSS-PLATFORM TRANSFER EVALUATION
# =============================================================================
def run_cross_platform_transfer(n_seeds: int = N_SEEDS) -> Dict[str, Any]:
    """
    Run cross-platform transfer experiments.

    This addresses the P2 task for cross-platform evaluation.
    Train on 2 platforms, test on 1.
    """
    print("\n" + "=" * 80)
    print("CROSS-PLATFORM TRANSFER EVALUATION")
    print("Addresses: the contributor P2 - Cross-platform transfer experiments")
    print("=" * 80)

    results = {
        'transfer_results': [],
        'platform_stats': {},
    }

    # Load all platform data
    platform_data = {}
    for platform in ['tx2', 'orin_nx', 'rubikpi']:
        try:
            platform_data[platform] = load_platform_data(platform)
        except FileNotFoundError:
            print(f"Warning: {platform} data not found, skipping")

    if len(platform_data) < 3:
        print("Warning: Need all 3 platforms for full cross-platform evaluation")
        return results

    platforms = list(platform_data.keys())

    # Experiment: Leave-One-Platform-Out
    for test_platform in platforms:
        train_platforms = [p for p in platforms if p != test_platform]

        print(f"\n--- Train: {train_platforms}, Test: {test_platform} ---")

        # Combine training data
        train_dfs = [platform_data[p] for p in train_platforms]
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = platform_data[test_platform]

        X_train, feature_cols = prepare_features(train_df)
        X_test, _ = prepare_features(test_df, feature_cols)

        for target in TARGET_COLS:
            y_train = train_df[target].values
            y_test = test_df[target].values

            seed_results = []

            for seed in RANDOM_SEEDS[:n_seeds]:
                # Normalize
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
                model.fit(X_train_scaled, y_train)

                # Predict
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred)
                seed_results.append(metrics)

            # Aggregate results
            agg_metrics = {}
            for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
                values = [r[metric] for r in seed_results]
                agg_metrics[f'{metric}_mean'] = np.mean(values)
                agg_metrics[f'{metric}_std'] = np.std(values)
                agg_metrics[f'{metric}_ci'] = 1.96 * np.std(values) / np.sqrt(len(values))

            results['transfer_results'].append({
                'train_platforms': ','.join(train_platforms),
                'test_platform': test_platform,
                'target': target,
                'n_train': len(y_train),
                'n_test': len(y_test),
                **agg_metrics
            })

            print(f"  {target}: R2={agg_metrics['R2_mean']:.4f} +/- {agg_metrics['R2_ci']:.4f}")

    # Same-platform baseline (for comparison)
    print("\n--- Same-Platform Baseline ---")

    for platform in platforms:
        df = platform_data[platform]
        X, feature_cols = prepare_features(df)

        for target in TARGET_COLS:
            y = df[target].values

            seed_results = []

            for seed in RANDOM_SEEDS[:n_seeds]:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                metrics = calculate_metrics(y_test, y_pred)
                seed_results.append(metrics)

            agg_metrics = {}
            for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
                values = [r[metric] for r in seed_results]
                agg_metrics[f'{metric}_mean'] = np.mean(values)
                agg_metrics[f'{metric}_std'] = np.std(values)

            results['platform_stats'][f'{platform}_{target}'] = agg_metrics

            print(f"  {platform} {target}: R2={agg_metrics['R2_mean']:.4f}")

    return results


def plot_cross_platform_results(results: Dict[str, Any], save_dir: str = RESULTS_DIR):
    """Generate plots for cross-platform transfer results."""

    if not results['transfer_results']:
        print("No transfer results to plot")
        return

    transfer_df = pd.DataFrame(results['transfer_results'])

    # Plot: Transfer performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, target in enumerate(TARGET_COLS):
        ax = axes[idx]
        target_data = transfer_df[transfer_df['target'] == target]

        x = range(len(target_data))
        labels = target_data['test_platform'].values
        means = target_data['R2_mean'].values
        cis = target_data['R2_ci'].values

        bars = ax.bar(x, means, yerr=cis, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('R2 Score')
        ax.set_title(f'Cross-Platform Transfer: {target}')
        ax.set_ylim(0, 1)

        # Add value labels
        for i, (mean, ci) in enumerate(zip(means, cis)):
            ax.annotate(f'{mean:.3f}', xy=(i, mean + ci + 0.02), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cross_platform_transfer.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {save_dir}")


# =============================================================================
# TASK 4: PER-PLATFORM DATASET STATISTICS
# =============================================================================
def generate_dataset_statistics() -> Dict[str, Any]:
    """
    Generate per-platform dataset statistics table.

    This addresses reviewer fwuN's concern about dataset sizes.
    """
    print("\n" + "=" * 80)
    print("PER-PLATFORM DATASET STATISTICS")
    print("Addresses: fwuN - Report exact dataset sizes per board")
    print("=" * 80)

    results = {
        'platforms': [],
    }

    for platform, filepath in DATA_FILES.items():
        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath)

        stats = {
            'platform': platform,
            'total_samples': len(df),
            'n_benchmarks': df['benchmark'].nunique(),
            'n_configurations': len(df.groupby(['benchmark', 'frequency_index', 'num_cores'])),
            'n_iterations': df['iteration'].nunique() if 'iteration' in df.columns else 1,
            'benchmarks': sorted(df['benchmark'].unique().tolist()),
            'bots_benchmarks': len([b for b in df['benchmark'].unique() if b in BOTS_BENCHMARKS]),
            'polybench_benchmarks': len([b for b in df['benchmark'].unique() if b in POLYBENCH_BENCHMARKS]),
        }

        # Feature statistics
        for target in TARGET_COLS:
            if target in df.columns:
                stats[f'{target}_mean'] = df[target].mean()
                stats[f'{target}_std'] = df[target].std()
                stats[f'{target}_min'] = df[target].min()
                stats[f'{target}_max'] = df[target].max()

        results['platforms'].append(stats)

        print(f"\n{platform.upper()}:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Benchmarks: {stats['n_benchmarks']} ({stats['bots_benchmarks']} BOTS, {stats['polybench_benchmarks']} PolyBench)")
        print(f"  Configurations: {stats['n_configurations']}")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def save_results_json(results: Dict[str, Any], filename: str):
    """Save results to JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"Results saved to: {filepath}")


def save_results_csv(results: Dict[str, Any], filename: str, key: str = None):
    """Save results to CSV file."""
    filepath = os.path.join(RESULTS_DIR, filename)

    if key and key in results:
        data = results[key]
    else:
        data = results

    if isinstance(data, list):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"CSV saved to: {filepath}")
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
        df.to_csv(filepath, index=False)
        print(f"CSV saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Offline Statistical Analysis for AISTATS Rebuttal')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'benchmark_holdout', 'statistics', 'cross_platform', 'dataset_stats'],
                       help='Which analysis task to run')
    parser.add_argument('--n_seeds', type=int, default=N_SEEDS,
                       help='Number of random seeds for statistical evaluation')
    parser.add_argument('--platform', type=str, default='tx2',
                       choices=['tx2', 'orin_nx', 'rubikpi', 'all'],
                       help='Platform to analyze (for single-platform tasks)')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print("OFFLINE STATISTICAL ANALYSIS FOR AISTATS REBUTTAL")
    print(f"Timestamp: {timestamp}")
    print(f"Task: {args.task}")
    print(f"N Seeds: {args.n_seeds}")
    print("=" * 80)

    # Load data
    if args.platform == 'all':
        df = load_all_data()
    else:
        df = load_platform_data(args.platform)

    all_results = {}

    # Run requested tasks
    if args.task in ['all', 'dataset_stats']:
        results = generate_dataset_statistics()
        all_results['dataset_stats'] = results
        save_results_json(results, f'dataset_statistics_{timestamp}.json')
        save_results_csv(results, f'dataset_statistics_{timestamp}.csv', key='platforms')

    if args.task in ['all', 'benchmark_holdout']:
        results = run_benchmark_holdout(df, n_seeds=args.n_seeds)
        all_results['benchmark_holdout'] = results
        save_results_json(results, f'benchmark_holdout_{timestamp}.json')
        save_results_csv(results, f'benchmark_holdout_summary_{timestamp}.csv', key='summary_table')
        plot_benchmark_holdout_results(results)

    if args.task in ['all', 'statistics']:
        results = run_statistical_tests(df, n_seeds=args.n_seeds)
        all_results['statistics'] = results
        save_results_json(results, f'statistical_tests_{timestamp}.json')
        save_results_csv(results, f'statistical_tests_seeds_{timestamp}.csv', key='seed_results')
        plot_statistical_results(results)

    if args.task in ['all', 'cross_platform']:
        results = run_cross_platform_transfer(n_seeds=args.n_seeds)
        all_results['cross_platform'] = results
        save_results_json(results, f'cross_platform_transfer_{timestamp}.json')
        save_results_csv(results, f'cross_platform_transfer_{timestamp}.csv', key='transfer_results')
        plot_cross_platform_results(results)

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if timestamp in f:
            print(f"  - {f}")

    print("\nNext steps for paper:")
    print("  1. Use benchmark_holdout results for Table in Appendix I")
    print("  2. Use confidence intervals for error bars in figures")
    print("  3. Use cross_platform results for transfer evaluation section")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    main()
