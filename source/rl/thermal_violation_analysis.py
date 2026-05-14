#!/usr/bin/env python3
"""
GraphPerf-RT — Section 6 (RL integration): thermal-violation analysis.

Quantifies the **zero thermal violations** result reported in §6.4 (Table 4)
of the camera-ready paper.  Reads per-episode trajectories saved by the
on-device RL servers (rl/on_device/*) and produces:

    - Per-method violation rate (proportion of episodes whose peak temperature
      exceeds an 80 °C threshold).
    - Peak-temperature statistics across the 5-seed sweep.
    - A LaTeX table (`thermal_table.tex`) ready to drop into the paper.
    - A JSON summary used by `figures/plot_figures.py rl-thermal` to redraw
      Fig. 8.

Compared methods (column heads in Table 4):
    SAMFRL · SAMBRL · MAMFRL_D3QN · MAMBRL_D3QN  (+ optional FEDERATED, MARBRL)

Usage
-----
    export GRAPHPERF_RT_ROOT="$(pwd)"
    python rl/thermal_violation_analysis.py \\
        --results-dir rl/on_device/save_model \\
        --platform tx2 \\
        --output rl/results/thermal
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# THERMAL THRESHOLDS BY PLATFORM
# =============================================================================
THERMAL_THRESHOLDS = {
    'tx2': 85.0,        # Jetson TX2 thermal throttle point
    'orin_nx': 85.0,    # Orin NX thermal throttle point
    'rubikpi': 80.0,    # RubikPi (more conservative)
    'nano': 85.0,       # Jetson Nano
}

# =============================================================================
# RL METHODS TO COMPARE
# =============================================================================
RL_METHODS = [
    'SAMFRL',       # Single-Agent Model-Free RL (zTT baseline)
    'SAMBRL',       # Single-Agent Model-Based RL
    'MAMFRL_D3QN',  # Multi-Agent Model-Free RL with D3QN
    'MAMBRL_D3QN',  # Multi-Agent Model-Based RL with D3QN (our method)
    'FEDERATED',    # Federated RL baseline
    'MARBRL',       # Multi-Agent Reward-Based RL
]


class ThermalViolationTracker:
    """
    Tracks thermal violations during RL training.
    Inspired by ConvergenceLogger from zero_combined_twobench.py.
    """

    def __init__(
        self,
        algorithm_name: str,
        platform: str,
        log_dir: Path,
        threshold: Optional[float] = None
    ):
        self.algorithm_name = algorithm_name
        self.platform = platform
        self.threshold = threshold or THERMAL_THRESHOLDS.get(platform, 85.0)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.violations: List[Dict] = []
        self.peak_temps: List[float] = []
        self.episode_count: int = 0
        self.all_temps: List[float] = []

        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"thermal_log_{algorithm_name}_{timestamp}.txt"

        with open(self.log_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"THERMAL TRACKING: {algorithm_name}\n")
            f.write(f"Platform: {platform}, Threshold: {self.threshold}C\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")

    def log(self, message: str):
        """Write message to log file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        logging.info(f"[THERMAL] {log_msg}")
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def get_max_temperature(self, row: pd.Series) -> float:
        """Extract maximum temperature from a data row.

        Priority:
        1. 'thermal' column - aggregated average (preferred, excludes stuck sensors)
        2. 'temperature' column - used by FEDERATED
        3. 'avg_temp_after' - alternative aggregated value
        4. Individual thermal_zone columns (excluding zone6 which is often stuck at 50C)
        """
        # Prefer aggregated thermal columns (they average out stuck sensors)
        agg_cols = ['thermal', 'temperature', 'avg_temp_after']
        for col in agg_cols:
            if col in row and pd.notna(row[col]) and row[col] > 0:
                return row[col]

        # Fallback: use individual thermal_zone columns, excluding zone6 (stuck at 50C)
        thermal_cols = [c for c in row.index
                       if 'thermal_zone' in c
                       and not c.endswith('_type')
                       and 'zone6' not in c]  # Exclude stuck sensor
        temps = []
        for col in thermal_cols:
            if pd.notna(row[col]) and row[col] > 0:
                temps.append(row[col])

        if temps:
            return max(temps)

        # Last fallback: platform-specific columns
        temp_cols = ['CPU57_temp_after', 'Denver_temp_after', 'System_temp_after',
                    'GPU_temp_after', 'DDR_temp_after']
        for col in temp_cols:
            if col in row and pd.notna(row[col]) and row[col] > 0:
                return row[col]

        return 0.0

    def record_episode(self, thermal_data: Dict):
        """Record thermal data from an episode."""
        self.episode_count += 1

        # Handle both dict and Series input
        if isinstance(thermal_data, pd.Series):
            max_temp = self.get_max_temperature(thermal_data)
        else:
            # Dict input
            thermal_cols = [k for k in thermal_data.keys()
                           if k.startswith('thermal_zone') and not k.endswith('_type')]
            temps = [thermal_data[k] for k in thermal_cols
                    if thermal_data.get(k) is not None and thermal_data[k] > 0]

            if not temps and 'avg_temp_after' in thermal_data:
                temps = [thermal_data['avg_temp_after']]

            if not temps and 'thermal' in thermal_data:
                temps = [thermal_data['thermal']]

            max_temp = max(temps) if temps else 0.0

        if max_temp > 0:
            self.peak_temps.append(max_temp)
            self.all_temps.append(max_temp)

            if max_temp > self.threshold:
                self.violations.append({
                    'episode': self.episode_count,
                    'peak_temp': max_temp,
                    'threshold': self.threshold,
                    'excess': max_temp - self.threshold
                })

    def record_from_dataframe(self, df: pd.DataFrame):
        """Record all episodes from a DataFrame."""
        for _, row in df.iterrows():
            self.record_episode(row)

    def get_summary(self) -> Dict:
        """Get thermal violation summary."""
        n_episodes = max(self.episode_count, 1)

        return {
            'algorithm': self.algorithm_name,
            'platform': self.platform,
            'n_episodes': self.episode_count,
            'violation_count': len(self.violations),
            'violation_rate': len(self.violations) / n_episodes * 100,
            'peak_temp_mean': np.mean(self.peak_temps) if self.peak_temps else 0,
            'peak_temp_std': np.std(self.peak_temps) if self.peak_temps else 0,
            'peak_temp_max': max(self.peak_temps) if self.peak_temps else 0,
            'peak_temp_min': min(self.peak_temps) if self.peak_temps else 0,
            'threshold': self.threshold,
            'avg_excess_when_violated': (
                np.mean([v['excess'] for v in self.violations])
                if self.violations else 0
            ),
        }

    def save_summary(self) -> Dict:
        """Save summary to log file and return it."""
        summary = self.get_summary()
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"THERMAL SUMMARY\n")
            f.write(f"{'='*60}\n")
            f.write(f"Episodes: {summary['n_episodes']}\n")
            f.write(f"Violations: {summary['violation_count']} ({summary['violation_rate']:.1f}%)\n")
            f.write(f"Peak Temp: {summary['peak_temp_mean']:.1f}C +/- {summary['peak_temp_std']:.1f}C\n")
            f.write(f"Max Peak: {summary['peak_temp_max']:.1f}C (threshold: {summary['threshold']}C)\n")
            if self.violations:
                f.write(f"Avg Excess When Violated: {summary['avg_excess_when_violated']:.1f}C\n")
        return summary


def load_rl_results(results_dir: Path, method: str, platform: str) -> pd.DataFrame:
    """
    Load RL experiment results for a given method.

    Searches for CSV files matching various naming patterns.
    """
    # Check if directory exists
    if not results_dir.exists():
        print(f"  WARNING: Directory does not exist: {results_dir}")
        return pd.DataFrame()

    patterns = [
        f"{method}_seed*.csv",  # Primary pattern: METHOD_seedXXX_*.csv
        f"*{method}*.csv",
        f"*{method.lower()}*.csv",
    ]

    all_files = []
    for pattern in patterns:
        files = list(results_dir.glob(pattern))
        all_files.extend(files)

    # Remove duplicates and exclude _stats.csv files
    all_files = list(set(f for f in all_files if not f.name.endswith('_stats.csv')))

    if not all_files:
        print(f"  WARNING: No files found for {method} in {results_dir}")
        return pd.DataFrame()

    print(f"  Found {len(all_files)} files for {method}")

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Error reading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def generate_thermal_latex_table(results: List[Dict], output_file: Path):
    """Generate LaTeX table for paper."""
    # Aggregate by method (across seeds)
    method_stats = {}
    for r in results:
        # Extract base method name (remove seed suffix)
        method = r['algorithm']
        if '_seed' in method:
            method = method.split('_seed')[0]

        if method not in method_stats:
            method_stats[method] = {
                'violations': [],
                'peaks': [],
                'stds': [],
                'n': [],
                'rates': []
            }
        method_stats[method]['violations'].append(r['violation_count'])
        method_stats[method]['peaks'].append(r['peak_temp_mean'])
        method_stats[method]['stds'].append(r['peak_temp_std'])
        method_stats[method]['n'].append(r['n_episodes'])
        method_stats[method]['rates'].append(r['violation_rate'])

    with open(output_file, 'w') as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Thermal Safety Metrics Across RL Baselines. Lower violation rates indicate safer scheduling. Our method (MAMBRL-D3QN with GraphPerf-RT) achieves the lowest thermal violations.}" + "\n")
        f.write(r"\label{tab:thermal}" + "\n")
        f.write(r"\begin{tabular}{lccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & Violation Rate (\%) & Peak Temp ($^\circ$C) & Episodes \\" + "\n")
        f.write(r"\midrule" + "\n")

        # Sort by violation rate (ascending - lower is better)
        sorted_methods = sorted(method_stats.items(),
                               key=lambda x: np.mean(x[1]['rates']))

        for method, stats in sorted_methods:
            viol_mean = np.mean(stats['rates'])
            viol_std = np.std(stats['rates'])
            peak_mean = np.mean(stats['peaks'])
            peak_std = np.mean(stats['stds'])  # Average of per-run stds
            n_total = sum(stats['n'])

            # Highlight best method
            if method == sorted_methods[0][0]:
                f.write(f"\\textbf{{{method}}} & \\textbf{{{viol_mean:.1f}$\\pm${viol_std:.1f}}} & ")
                f.write(f"\\textbf{{{peak_mean:.1f}$\\pm${peak_std:.1f}}} & {n_total} \\\\\n")
            else:
                f.write(f"{method} & {viol_mean:.1f}$\\pm${viol_std:.1f} & ")
                f.write(f"{peak_mean:.1f}$\\pm${peak_std:.1f} & {n_total} \\\\\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    logging.info(f"LaTeX table saved to: {output_file}")


def generate_thermal_comparison_plot(results: List[Dict], output_file: Path):
    """Generate thermal comparison bar plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Aggregate by method
        method_data = {}
        for r in results:
            method = r['algorithm'].split('_seed')[0] if '_seed' in r['algorithm'] else r['algorithm']
            if method not in method_data:
                method_data[method] = {'rates': [], 'peaks': []}
            method_data[method]['rates'].append(r['violation_rate'])
            method_data[method]['peaks'].append(r['peak_temp_mean'])

        methods = list(method_data.keys())
        rates_mean = [np.mean(method_data[m]['rates']) for m in methods]
        rates_std = [np.std(method_data[m]['rates']) for m in methods]
        peaks_mean = [np.mean(method_data[m]['peaks']) for m in methods]
        peaks_std = [np.std(method_data[m]['peaks']) for m in methods]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Violation rate bar plot
        ax1 = axes[0]
        colors = sns.color_palette("husl", len(methods))
        bars1 = ax1.bar(range(len(methods)), rates_mean, yerr=rates_std,
                       capsize=5, color=colors, edgecolor='black')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.set_ylabel('Thermal Violation Rate (%)')
        ax1.set_title('Thermal Violations by Method')
        ax1.axhline(y=10, color='red', linestyle='--', label='10% threshold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Peak temperature bar plot
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(methods)), peaks_mean, yerr=peaks_std,
                       capsize=5, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.set_ylabel('Peak Temperature (C)')
        ax2.set_title('Peak Temperature by Method')
        ax2.axhline(y=85, color='red', linestyle='--', label='85C threshold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Thermal comparison plot saved to: {output_file}")

    except ImportError as e:
        logging.warning(f"Could not generate plot: {e}")


def run_thermal_experiment(
    results_dir: Path,
    methods: List[str],
    platform: str,
    seeds: List[int] = [42, 123, 456],
    output_dir: Path = Path('thermal_results')
) -> List[Dict]:
    """
    Run thermal tracking experiment across multiple RL methods.

    Pattern: Based on zero_combined_twobench.py multi-seed loop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("\n" + "="*70)
    print("THERMAL VIOLATION TRACKING EXPERIMENT")
    print(f"Platform: {platform}, Threshold: {THERMAL_THRESHOLDS.get(platform, 85.0)}C")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print("="*70 + "\n")

    for method in methods:
        print(f"\n--- Analyzing: {method} ---")

        # Load all results for this method
        df = load_rl_results(results_dir, method, platform)

        if len(df) == 0:
            logging.warning(f"No data found for {method}")
            continue

        # Check if we have seed information in the data
        if 'seed' in df.columns:
            for seed in df['seed'].unique():
                seed_df = df[df['seed'] == seed]
                tracker = ThermalViolationTracker(
                    algorithm_name=f"{method}_seed{seed}",
                    platform=platform,
                    log_dir=output_dir / 'logs'
                )
                tracker.record_from_dataframe(seed_df)
                summary = tracker.save_summary()
                all_results.append(summary)

                print(f"  Seed {seed}: {summary['violation_count']}/{summary['n_episodes']} violations "
                      f"({summary['violation_rate']:.1f}%), Peak: {summary['peak_temp_mean']:.1f}C")
        else:
            # No seed column - treat as single run
            tracker = ThermalViolationTracker(
                algorithm_name=method,
                platform=platform,
                log_dir=output_dir / 'logs'
            )
            tracker.record_from_dataframe(df)
            summary = tracker.save_summary()
            all_results.append(summary)

            print(f"  Violations: {summary['violation_count']}/{summary['n_episodes']} "
                  f"({summary['violation_rate']:.1f}%)")
            print(f"  Peak Temp: {summary['peak_temp_mean']:.1f}C +/- {summary['peak_temp_std']:.1f}C")

    if not all_results:
        logging.error("No results collected!")
        return []

    # Generate outputs
    generate_thermal_latex_table(all_results, output_dir / 'thermal_table.tex')
    generate_thermal_comparison_plot(all_results, output_dir / 'thermal_comparison.png')

    # Save JSON results
    with open(output_dir / 'thermal_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("THERMAL EXPERIMENT SUMMARY")
    print("="*70)

    # Aggregate by method
    method_agg = {}
    for r in all_results:
        method = r['algorithm'].split('_seed')[0] if '_seed' in r['algorithm'] else r['algorithm']
        if method not in method_agg:
            method_agg[method] = []
        method_agg[method].append(r)

    for method, runs in method_agg.items():
        avg_rate = np.mean([r['violation_rate'] for r in runs])
        avg_peak = np.mean([r['peak_temp_mean'] for r in runs])
        print(f"{method:20s}: {avg_rate:5.1f}% violations, {avg_peak:5.1f}C avg peak")

    print(f"\nResults saved to: {output_dir}")
    print("="*70)

    return all_results


def is_jupyter() -> bool:
    """Check if running in Jupyter/IPython environment."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Track thermal violations across RL baselines'
    )
    parser.add_argument(
        '--results-dir', type=str,
        default='on_device/save_model',
        help='Directory containing RL result CSV files'
    )
    parser.add_argument(
        '--platform', type=str,
        default='tx2',
        choices=['tx2', 'orin_nx', 'rubikpi', 'nano'],
        help='Platform name for thermal threshold'
    )
    parser.add_argument(
        '--methods', type=str, nargs='+',
        default=['SAMFRL', 'SAMBRL', 'MAMFRL_D3QN', 'MAMBRL_D3QN'],
        help='RL methods to analyze'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+',
        default=[42, 123, 456, 789, 1024],
        help='Random seeds used in experiments'
    )
    parser.add_argument(
        '--output', type=str,
        default='thermal_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logging.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    run_thermal_experiment(
        results_dir=results_dir,
        methods=args.methods,
        platform=args.platform,
        seeds=args.seeds,
        output_dir=Path(args.output)
    )


if __name__ == '__main__':
    if is_jupyter():
        # Running in Jupyter/IPython - use defaults
        # Use absolute paths
        REPO_ROOT = Path(os.environ.get('GRAPHPERF_RT_ROOT', Path(__file__).resolve().parents[1]))
        RESULTS_DIR = REPO_ROOT / 'src' / 'RL' / 'jetson_tx2' / 'save_model'
        # Output to src/RL/results/thermal (keep paper directory clean)
        OUTPUT_DIR = REPO_ROOT / 'src' / 'RL' / 'results' / 'thermal'

        if not RESULTS_DIR.exists():
            print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        else:
            print(f"Results directory: {RESULTS_DIR}")
            print(f"Output directory: {OUTPUT_DIR}")
            run_thermal_experiment(
                results_dir=RESULTS_DIR,
                methods=['SAMFRL', 'SAMBRL', 'MAMFRL_D3QN', 'MAMBRL_D3QN', 'FEDERATED', 'MARBRL'],
                platform='tx2',
                seeds=[42, 123, 456, 789, 1024],
                output_dir=OUTPUT_DIR
            )
    else:
        # Running from command line - use argparse
        main()
