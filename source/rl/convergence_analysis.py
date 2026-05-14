#!/usr/bin/env python3
"""
convergence_comparison.py - Compare Convergence Across RL Methods

Analyzes and compares convergence behavior of different RL methods:
- MAMBRL with GraphPerf-RT world model
- MAMBRL with FCN environment model
- MAMFRL (model-free baseline)
- SAMFRL (single-agent model-free)

Based on ConvergenceLogger from zero_combined_twobench.py.

Purpose:
- Detect convergence epochs for each method
- Compare convergence speed (sample efficiency)
- Generate convergence plots for paper
- Support claim that GraphPerf-RT accelerates learning

Usage:
    python convergence_comparison.py --results-dir results/ --output convergence_analysis
    python convergence_comparison.py --csv-files file1.csv file2.csv --output analysis
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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# =============================================================================
# CONVERGENCE LOGGER (from zero_combined_twobench.py)
# =============================================================================
class ConvergenceLogger:
    """
    Logs and detects convergence based on:
    1. Makespan becoming steady (low std deviation over window)
    2. Frequency stabilizing at max (11) since makespan is primary objective

    Adapted from zero_combined_twobench.py with additional features:
    - Multi-method comparison
    - Confidence interval calculation
    - Publication-quality plot generation
    """

    def __init__(
        self,
        algorithm_name: str,
        log_dir: Path,
        window_size: int = 10,
        makespan_threshold: float = 0.05,
        freq_max: int = 11
    ):
        """
        Args:
            algorithm_name: Name of algorithm (e.g., 'MAMBRL_GraphPerf')
            log_dir: Directory to save convergence logs
            window_size: Number of epochs to check for stability
            makespan_threshold: Max relative std dev for makespan to be "steady"
            freq_max: Maximum frequency index (11 for Jetson TX2)
        """
        self.algorithm_name = algorithm_name
        self.log_dir = Path(log_dir)
        self.window_size = window_size
        self.makespan_threshold = makespan_threshold
        self.freq_max = freq_max

        self.convergence_epoch: Optional[int] = None
        self.converged: bool = False
        self.convergence_details: List[Dict] = []

        # Setup logging file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"convergence_log_{algorithm_name}_{timestamp}.txt"

        with open(self.log_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"CONVERGENCE LOG: {algorithm_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Window size: {window_size}, Makespan threshold: {makespan_threshold}\n")
            f.write(f"Max frequency: {freq_max}\n")
            f.write(f"{'='*60}\n\n")

    def log(self, message: str):
        """Write message to log file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        logging.info(f"[CONVERGENCE] {log_msg}")
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def analyze_csv(self, csv_path: Path) -> Optional[Dict]:
        """
        Analyze CSV file to detect convergence point.

        Returns:
            dict with convergence info: epoch, makespan_at_convergence, freq_stability
        """
        if not csv_path.exists():
            self.log(f"ERROR: CSV file not found: {csv_path}")
            return None

        self.log(f"Analyzing: {csv_path.name}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.log(f"ERROR: Failed to read CSV: {e}")
            return None

        # Determine iteration column
        iter_col = None
        for col in ['iteration', 'epoch', 'episode', 'step']:
            if col in df.columns:
                iter_col = col
                break

        if iter_col is None:
            # Use row index as iteration
            df['iteration'] = range(len(df))
            iter_col = 'iteration'

        # Get makespan column
        makespan_col = None
        for col in ['makespan', 'time_elapsed', 'execution_time']:
            if col in df.columns:
                makespan_col = col
                break

        if makespan_col is None:
            self.log("ERROR: No makespan column found")
            return None

        # Get frequency column
        freq_col = None
        for col in ['freq', 'frequency', 'freq_idx']:
            if col in df.columns:
                freq_col = col
                break

        # Aggregate by iteration
        agg_dict = {makespan_col: 'sum'}
        if freq_col:
            agg_dict[freq_col] = 'first'

        agg_df = df.groupby(iter_col).agg(agg_dict).reset_index()

        makespans = agg_df[makespan_col].values
        freqs = agg_df[freq_col].values if freq_col else np.zeros(len(makespans))

        self.log(f"Total epochs: {len(makespans)}")
        self.log(f"Makespan range: {makespans.min():.3f} - {makespans.max():.3f}")
        if freq_col:
            self.log(f"Frequency range: {freqs.min()} - {freqs.max()}")

        # Detect convergence: sliding window analysis
        convergence_epoch = None

        for i in range(self.window_size, len(makespans)):
            window_start = i - self.window_size
            window_makespans = makespans[window_start:i]
            window_freqs = freqs[window_start:i]

            # Check makespan stability (relative std dev)
            mean_makespan = np.mean(window_makespans)
            std_makespan = np.std(window_makespans)
            rel_std = std_makespan / (mean_makespan + 1e-10)
            makespan_steady = rel_std < self.makespan_threshold

            # Check frequency at max
            freq_at_max_ratio = np.mean(window_freqs == self.freq_max) if freq_col else 1.0
            freq_stable_at_max = freq_at_max_ratio >= 0.8

            # Store analysis details
            self.convergence_details.append({
                'epoch': i,
                'mean_makespan': mean_makespan,
                'rel_std': rel_std,
                'makespan_steady': makespan_steady,
                'freq_at_max_ratio': freq_at_max_ratio,
                'freq_stable': freq_stable_at_max
            })

            # Convergence: makespan steady AND freq at max (or no freq data)
            if makespan_steady and freq_stable_at_max and convergence_epoch is None:
                convergence_epoch = i
                self.converged = True
                self.convergence_epoch = i
                self.log(f"\n*** CONVERGENCE DETECTED at epoch {i} ***")
                self.log(f"    Makespan: {mean_makespan:.3f}s (rel_std={rel_std:.4f})")
                if freq_col:
                    self.log(f"    Frequency: {freq_at_max_ratio*100:.1f}% at max ({self.freq_max})")

        # Summary
        self.log(f"\n{'='*40}")
        self.log(f"CONVERGENCE SUMMARY for {self.algorithm_name}")
        self.log(f"{'='*40}")

        result = {
            'algorithm': self.algorithm_name,
            'csv_file': str(csv_path),
            'converged': self.converged,
            'convergence_epoch': convergence_epoch,
            'total_epochs': len(makespans),
            'final_makespan': makespans[-1],
            'final_freq': freqs[-1] if freq_col else None,
            'mean_makespan': np.mean(makespans[-self.window_size:]),
            'std_makespan': np.std(makespans[-self.window_size:]),
            'makespans': makespans.tolist(),
            'log_file': str(self.log_file)
        }

        if convergence_epoch is not None:
            self.log(f"CONVERGED at epoch: {convergence_epoch}")
            self.log(f"Final makespan (avg last {self.window_size}): {result['mean_makespan']:.3f}s")
        else:
            self.log(f"NOT CONVERGED within {len(makespans)} epochs")
            self.log(f"Final makespan: {makespans[-1]:.3f}s")

        return result


# =============================================================================
# CONVERGENCE COMPARISON
# =============================================================================
class ConvergenceComparison:
    """
    Compares convergence behavior across multiple RL methods.

    Generates:
    - Convergence epoch comparison
    - Learning curve plots
    - Statistical analysis
    - LaTeX tables for paper
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, List[Dict]] = {}

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"convergence_comparison_{timestamp}.log"

    def log(self, message: str):
        """Write to log file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        logging.info(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def analyze_method(
        self,
        method_name: str,
        csv_files: List[Path],
        window_size: int = 10,
        makespan_threshold: float = 0.05
    ) -> List[Dict]:
        """Analyze convergence for a single method across multiple runs."""
        self.log(f"\n--- Analyzing: {method_name} ({len(csv_files)} files) ---")

        method_results = []

        for csv_file in csv_files:
            logger = ConvergenceLogger(
                algorithm_name=method_name,
                log_dir=self.output_dir / 'logs',
                window_size=window_size,
                makespan_threshold=makespan_threshold
            )

            result = logger.analyze_csv(csv_file)
            if result:
                method_results.append(result)

        self.results[method_name] = method_results
        return method_results

    def compare_all(self) -> Dict:
        """Generate comparison statistics across all methods."""
        comparison = {}

        self.log("\n" + "="*70)
        self.log("CONVERGENCE COMPARISON SUMMARY")
        self.log("="*70)

        for method, results in self.results.items():
            if not results:
                continue

            conv_epochs = [r['convergence_epoch'] for r in results
                          if r.get('convergence_epoch') is not None]
            final_makespans = [r['final_makespan'] for r in results
                              if r.get('final_makespan') is not None]
            converged_count = sum(1 for r in results if r.get('converged', False))

            stats = {
                'method': method,
                'n_runs': len(results),
                'n_converged': converged_count,
                'convergence_rate': converged_count / len(results) * 100,
            }

            if conv_epochs:
                stats['conv_epoch_mean'] = np.mean(conv_epochs)
                stats['conv_epoch_std'] = np.std(conv_epochs)
                stats['conv_epoch_min'] = min(conv_epochs)
                stats['conv_epoch_max'] = max(conv_epochs)

            if final_makespans:
                stats['makespan_mean'] = np.mean(final_makespans)
                stats['makespan_std'] = np.std(final_makespans)

            comparison[method] = stats

            # Log summary
            self.log(f"\n{method}:")
            self.log(f"  Runs: {stats['n_runs']}, Converged: {stats['n_converged']} "
                    f"({stats['convergence_rate']:.0f}%)")
            if conv_epochs:
                self.log(f"  Conv. Epoch: {stats['conv_epoch_mean']:.1f} +/- {stats['conv_epoch_std']:.1f}")
            if final_makespans:
                self.log(f"  Final Makespan: {stats['makespan_mean']:.3f}s +/- {stats['makespan_std']:.3f}s")

        return comparison

    def generate_convergence_plot(self):
        """Generate convergence comparison plot."""
        if not MATPLOTLIB_AVAILABLE:
            self.log("WARNING: matplotlib not available, skipping plot generation")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Colors for methods
        colors = sns.color_palette("husl", len(self.results))

        # Plot 1: Learning curves
        ax1 = axes[0]
        for idx, (method, results) in enumerate(self.results.items()):
            # Aggregate makespans across runs
            all_makespans = []
            for r in results:
                if 'makespans' in r:
                    all_makespans.append(r['makespans'])

            if not all_makespans:
                continue

            # Pad to same length
            max_len = max(len(m) for m in all_makespans)
            padded = []
            for m in all_makespans:
                if len(m) < max_len:
                    m = list(m) + [m[-1]] * (max_len - len(m))
                padded.append(m[:max_len])

            makespans_array = np.array(padded)
            mean_makespan = np.mean(makespans_array, axis=0)
            std_makespan = np.std(makespans_array, axis=0)

            epochs = np.arange(len(mean_makespan))

            ax1.plot(epochs, mean_makespan, color=colors[idx], label=method, linewidth=2)
            ax1.fill_between(epochs,
                            mean_makespan - std_makespan,
                            mean_makespan + std_makespan,
                            color=colors[idx], alpha=0.2)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Makespan (s)')
        ax1.set_title('Learning Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Convergence epoch comparison
        ax2 = axes[1]
        methods = []
        conv_epochs_mean = []
        conv_epochs_std = []

        comparison = self.compare_all()
        for method, stats in comparison.items():
            if 'conv_epoch_mean' in stats:
                methods.append(method)
                conv_epochs_mean.append(stats['conv_epoch_mean'])
                conv_epochs_std.append(stats['conv_epoch_std'])

        if methods:
            x = np.arange(len(methods))
            bars = ax2.bar(x, conv_epochs_mean, yerr=conv_epochs_std,
                          capsize=5, color=colors[:len(methods)], edgecolor='black')
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods, rotation=45, ha='right')
            ax2.set_ylabel('Convergence Epoch')
            ax2.set_title('Convergence Speed Comparison')
            ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"Plot saved to: {self.output_dir / 'convergence_comparison.png'}")

    def generate_latex_table(self):
        """Generate LaTeX table for paper."""
        comparison = self.compare_all()

        table_file = self.output_dir / 'convergence_table.tex'

        with open(table_file, 'w') as f:
            f.write(r"\begin{table}[t]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{Convergence Comparison Across RL Methods. GraphPerf-RT world model accelerates convergence by $\sim$X$\times$ compared to FCN environment model.}" + "\n")
            f.write(r"\label{tab:convergence}" + "\n")
            f.write(r"\begin{tabular}{lcccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Method & Conv. Rate (\%) & Conv. Epoch & Final Makespan (s) & Speedup \\" + "\n")
            f.write(r"\midrule" + "\n")

            # Sort by convergence epoch (lower is better)
            sorted_methods = sorted(
                comparison.items(),
                key=lambda x: x[1].get('conv_epoch_mean', float('inf'))
            )

            # Calculate speedup relative to slowest converging method
            baseline_epoch = max(
                s.get('conv_epoch_mean', 0) for m, s in sorted_methods
                if s.get('conv_epoch_mean') is not None
            )

            for method, stats in sorted_methods:
                conv_rate = stats.get('convergence_rate', 0)
                conv_epoch = stats.get('conv_epoch_mean')
                conv_std = stats.get('conv_epoch_std', 0)
                makespan = stats.get('makespan_mean')
                makespan_std = stats.get('makespan_std', 0)

                if conv_epoch and baseline_epoch:
                    speedup = baseline_epoch / conv_epoch
                else:
                    speedup = None

                # Format values
                conv_str = f"{conv_epoch:.0f}$\\pm${conv_std:.0f}" if conv_epoch else "--"
                makespan_str = f"{makespan:.3f}$\\pm${makespan_std:.3f}" if makespan else "--"
                speedup_str = f"{speedup:.1f}$\\times$" if speedup else "--"

                # Highlight best method
                if method == sorted_methods[0][0] and conv_epoch:
                    f.write(f"\\textbf{{{method}}} & \\textbf{{{conv_rate:.0f}}} & ")
                    f.write(f"\\textbf{{{conv_str}}} & \\textbf{{{makespan_str}}} & ")
                    f.write(f"\\textbf{{{speedup_str}}} \\\\\n")
                else:
                    f.write(f"{method} & {conv_rate:.0f} & {conv_str} & ")
                    f.write(f"{makespan_str} & {speedup_str} \\\\\n")

            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")

        self.log(f"LaTeX table saved to: {table_file}")

    def save_results(self):
        """Save all results to JSON."""
        results_file = self.output_dir / 'convergence_results.json'

        # Convert results to serializable format
        serializable_results = {}
        for method, results in self.results.items():
            serializable_results[method] = []
            for r in results:
                # Remove non-serializable items
                clean_r = {k: v for k, v in r.items()
                          if k != 'makespans' or isinstance(v, list)}
                serializable_results[method].append(clean_r)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.log(f"Results saved to: {results_file}")


# =============================================================================
# MAIN
# =============================================================================
def find_csv_files(results_dir: Path, method: str) -> List[Path]:
    """Find CSV files for a given method."""
    patterns = [
        f"*{method}*.csv",
        f"*{method.lower()}*.csv",
        f"server_{method.lower()}_*.csv",
    ]

    files = []
    for pattern in patterns:
        files.extend(results_dir.glob(pattern))

    return list(set(files))


def main():
    parser = argparse.ArgumentParser(
        description='Compare convergence across RL methods'
    )
    parser.add_argument(
        '--results-dir', type=str,
        default='results',
        help='Directory containing result CSV files'
    )
    parser.add_argument(
        '--csv-files', type=str, nargs='+',
        help='Specific CSV files to analyze (overrides --results-dir)'
    )
    parser.add_argument(
        '--methods', type=str, nargs='+',
        default=['MAMBRL_GraphPerf', 'MAMBRL_FCN', 'MAMFRL', 'SAMFRL'],
        help='Methods to compare'
    )
    parser.add_argument(
        '--window-size', type=int,
        default=10,
        help='Window size for stability detection'
    )
    parser.add_argument(
        '--threshold', type=float,
        default=0.05,
        help='Makespan stability threshold (relative std)'
    )
    parser.add_argument(
        '--output', type=str,
        default='convergence_analysis',
        help='Output directory'
    )

    args = parser.parse_args()

    comparison = ConvergenceComparison(output_dir=Path(args.output))

    comparison.log("="*70)
    comparison.log("CONVERGENCE COMPARISON ANALYSIS")
    comparison.log(f"Methods: {args.methods}")
    comparison.log("="*70)

    if args.csv_files:
        # Use specific files
        for csv_file in args.csv_files:
            csv_path = Path(csv_file)
            # Try to extract method name from filename
            method_name = csv_path.stem.split('_')[1] if '_' in csv_path.stem else 'Unknown'
            comparison.analyze_method(
                method_name=method_name,
                csv_files=[csv_path],
                window_size=args.window_size,
                makespan_threshold=args.threshold
            )
    else:
        # Find files by method
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            logging.error(f"Results directory not found: {results_dir}")
            sys.exit(1)

        for method in args.methods:
            csv_files = find_csv_files(results_dir, method)
            if csv_files:
                comparison.analyze_method(
                    method_name=method,
                    csv_files=csv_files,
                    window_size=args.window_size,
                    makespan_threshold=args.threshold
                )
            else:
                comparison.log(f"No files found for {method}")

    # Generate outputs
    comparison.compare_all()
    comparison.generate_convergence_plot()
    comparison.generate_latex_table()
    comparison.save_results()

    print(f"\nConvergence analysis complete. Results saved to: {args.output}")


if __name__ == '__main__':
    main()
