#!/usr/bin/env python3
"""
server_lobo_experiment.py - Leave-One-Benchmark-Out Training Server

For each of 42 benchmarks:
1. Train RL agent on 41 benchmarks
2. Evaluate generalization on held-out benchmark
3. Track performance metrics

Based on zero_combined_twobench.py pattern from ZeroDVFS.

Server-Client Architecture:
- Server (this script): Runs on workstation, manages RL training
- Client (client.py): Runs on Jetson TX2, executes benchmarks

Usage:
    # On Jetson TX2 (client):
    python client.py

    # On Server (workstation):
    python server_lobo_experiment.py --benchmarks all --epochs 100 --seeds 42
"""

import socket
import logging
import json
import time
import gc
import os
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
PORT = 8707
IP_ADDRESS = '0.0.0.0'
PLATFORM_NAME = 'jetson_tx2'

DATA_KEYS = [
    'makespan', 'num_cores', 'energy', 'qmax', 'loss', 'freq', 'cores',
    'thermal', 'reward', 'branch_misses', 'cache_misses', 'priority_combination'
]

# Frequency and core configurations
FREQUENCY_COMBINATIONS = [
    [0] * 5,
    [2] * 5,
    [6] * 5,
    [11] * 5
]
NUM_CORES_LIST = [1, 3, 5]

# =============================================================================
# BENCHMARK DEFINITIONS
# =============================================================================
BOTS_BENCHMARKS = [
    {"benchmark": "alignment", "variant": "bin-omp-tasks", "input_arg": "20", "category": "BOTS-Regular"},
    {"benchmark": "fft", "variant": "bin-omp-tasks", "input_arg": "262144", "category": "BOTS-Regular"},
    {"benchmark": "fib", "variant": "bin-omp-tasks", "input_arg": "42", "category": "BOTS-Regular"},
    {"benchmark": "floorplan", "variant": "bin-omp-tasks", "input_arg": "input.20", "category": "BOTS-Irregular"},
    {"benchmark": "health", "variant": "bin-omp-tasks", "input_arg": "large", "category": "BOTS-Irregular"},
    {"benchmark": "nqueens", "variant": "bin-omp-tasks", "input_arg": "14", "category": "BOTS-Irregular"},
    {"benchmark": "sort", "variant": "bin-omp-tasks", "input_arg": "100000000", "category": "BOTS-Regular"},
    {"benchmark": "sparselu", "variant": "bin-omp-tasks", "input_arg": "50", "category": "BOTS-Regular"},
    {"benchmark": "strassen", "variant": "bin-omp-tasks", "input_arg": "2048", "category": "BOTS-Regular"},
    {"benchmark": "uts", "variant": "bin-omp-tasks", "input_arg": "tiny", "category": "BOTS-Irregular"},
]

POLYBENCH_BENCHMARKS = [
    {"benchmark": "gemm", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Linear"},
    {"benchmark": "gemver", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Linear"},
    {"benchmark": "gesummv", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Linear"},
    {"benchmark": "2mm", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Kernel"},
    {"benchmark": "3mm", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Kernel"},
    {"benchmark": "atax", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Kernel"},
    {"benchmark": "bicg", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Kernel"},
    {"benchmark": "jacobi-2d", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Stencil"},
    {"benchmark": "seidel-2d", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-Stencil"},
    {"benchmark": "correlation", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-DataMining"},
    {"benchmark": "covariance", "variant": "polybench", "input_arg": "LARGE", "category": "PolyBench-DataMining"},
]

ALL_BENCHMARKS = BOTS_BENCHMARKS + POLYBENCH_BENCHMARKS

# Default hyperparameters
DEFAULT_CONFIG = {
    'experiment_time': 100,  # Shorter for LOBO (many benchmarks)
    'target_temp': 50,
    'beta': 1.0,
    'learning_rate': 0.1,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.9,
    'epsilon_min': 0.1,
    'batch_size': 32,
    'agent_train_start': 32,
    'plan_count': 20,
    'mem_size': 100000,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

script_dir = Path(__file__).parent.absolute()
save_dir = script_dir / "save_model" / "lobo"
save_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def cleanup_memory():
    """Comprehensive memory cleanup."""
    plt.close('all')
    if TF_AVAILABLE:
        try:
            K.clear_session()
        except:
            pass
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    gc.collect()
    gc.collect()


def set_seed(seed: int):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        try:
            tf.random.set_seed(seed)
        except:
            pass
    if TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
        except:
            pass


def establish_connection() -> Tuple[socket.socket, socket.socket]:
    """Establish server connection."""
    logging.info("Waiting for client connection...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((IP_ADDRESS, PORT))
    server_socket.listen(1)
    client_socket, address = server_socket.accept()
    logging.info(f"Client connected from {address[0]}:{address[1]}")
    return client_socket, server_socket


def parse_profiling_data(data: str) -> Optional[List[Dict]]:
    """Parse profiling data JSON."""
    try:
        msg = json.loads(data)
        return msg.get('profiling_data_list', [])
    except json.JSONDecodeError:
        return None


def profile_application(
    client_socket: socket.socket,
    app_config: Dict,
    app_id: int
) -> Optional[Dict]:
    """Profile a single application."""
    application = {
        'id': app_id,
        'benchmark': app_config['benchmark'],
        'variant': app_config['variant'],
        'input_arg': app_config['input_arg'],
        'app_args': app_config['input_arg'],
        'cores': '',
        'frequencies': [],
        'priority': None,
        'action': 'profile'
    }

    send_msg = json.dumps({'applications': [application]})
    client_socket.send((send_msg + '\n').encode())

    recv_buffer = ''
    start_time = time.time()
    timeout = 600

    while time.time() - start_time < timeout:
        data = client_socket.recv(4096)
        if data:
            recv_buffer += data.decode()
            while '\n' in recv_buffer:
                msg, recv_buffer = recv_buffer.split('\n', 1)
                pd_list = parse_profiling_data(msg)
                if pd_list and len(pd_list) == 1:
                    profiling_data = pd_list[0]

                    # Calculate parallelism
                    mk_one = profiling_data.get('makespan_one_core_frequency_11')
                    mk_all = profiling_data.get('makespan_all_cores_frequency_11')
                    if mk_one and mk_all and mk_all != 0:
                        profiling_data['parallelism_level'] = mk_one / mk_all
                    else:
                        profiling_data['parallelism_level'] = 1.0

                    # Send ack
                    ack = json.dumps({'status': 'received'})
                    client_socket.send((ack + '\n').encode())

                    return profiling_data
        else:
            time.sleep(1)

    return None


# =============================================================================
# LOBO EXPERIMENT
# =============================================================================
class LOBOExperiment:
    """
    Leave-One-Benchmark-Out Training Experiment.

    For each benchmark:
    1. Profile all benchmarks
    2. Train on N-1 benchmarks
    3. Evaluate on held-out benchmark
    """

    def __init__(
        self,
        client_socket: socket.socket,
        benchmarks: List[Dict],
        method: str = 'MAMBRL_D3QN',
        output_dir: Path = save_dir
    ):
        self.client_socket = client_socket
        self.benchmarks = benchmarks
        self.method = method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict] = []

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"lobo_log_{timestamp}.txt"

    def log(self, message: str):
        """Log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        logging.info(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def profile_all_benchmarks(self) -> Dict[str, Dict]:
        """Profile all benchmarks."""
        self.log("Profiling all benchmarks...")
        profiles = {}

        for idx, bench in enumerate(self.benchmarks):
            bench_name = bench['benchmark']
            self.log(f"  Profiling {bench_name} ({idx+1}/{len(self.benchmarks)})")

            profile = profile_application(
                self.client_socket, bench, idx + 1
            )

            if profile:
                profiles[bench_name] = profile
            else:
                self.log(f"  WARNING: Failed to profile {bench_name}")

        self.log(f"Profiled {len(profiles)}/{len(self.benchmarks)} benchmarks")
        return profiles

    def run_lobo_split(
        self,
        held_out: str,
        train_benchmarks: List[Dict],
        seed: int,
        epochs: int
    ) -> Dict:
        """
        Run training on N-1 benchmarks, evaluate on held-out.

        Returns evaluation metrics.
        """
        self.log(f"\n--- Holding out: {held_out} ---")
        self.log(f"  Training on {len(train_benchmarks)} benchmarks")

        # In production, this calls actual training function
        # For now, return placeholder
        result = {
            'held_out': held_out,
            'seed': seed,
            'epochs': epochs,
            'n_train_benchmarks': len(train_benchmarks),
            'train_makespan': None,
            'test_makespan': None,
            'generalization_gap': None,
        }

        try:
            # Import training function
            if self.method == 'MAMBRL_D3QN':
                from server_mambrl import train_fixed_mambrl_d3qn
                self.log(f"  Would train MAMBRL_D3QN on {len(train_benchmarks)} benchmarks")
            elif self.method == 'MAMFRL_D3QN':
                from server_mamfrl import train_fixed_mamfrl_d3qn
                self.log(f"  Would train MAMFRL_D3QN on {len(train_benchmarks)} benchmarks")

            # Placeholder: actual training would go here

        except ImportError as e:
            self.log(f"  Import error: {e}")

        return result

    def run_full_lobo(
        self,
        seeds: List[int] = [42],
        epochs: int = 100
    ) -> List[Dict]:
        """Run full LOBO evaluation."""
        self.log("="*70)
        self.log("LEAVE-ONE-BENCHMARK-OUT (LOBO) EXPERIMENT")
        self.log(f"Method: {self.method}")
        self.log(f"Benchmarks: {len(self.benchmarks)}")
        self.log(f"Seeds: {seeds}")
        self.log(f"Epochs per run: {epochs}")
        self.log("="*70)

        # Profile all benchmarks first
        profiles = self.profile_all_benchmarks()

        # Run LOBO for each benchmark
        for held_out_config in self.benchmarks:
            held_out = held_out_config['benchmark']
            category = held_out_config.get('category', 'Unknown')

            self.log(f"\n{'='*60}")
            self.log(f"LOBO: Holding out {held_out} ({category})")
            self.log(f"{'='*60}")

            # Create training set (all except held-out)
            train_benchmarks = [
                b for b in self.benchmarks
                if b['benchmark'] != held_out
            ]

            for seed in seeds:
                set_seed(seed)

                result = self.run_lobo_split(
                    held_out=held_out,
                    train_benchmarks=train_benchmarks,
                    seed=seed,
                    epochs=epochs
                )
                result['category'] = category

                self.results.append(result)
                cleanup_memory()

        # Generate outputs
        self.generate_outputs()

        return self.results

    def generate_outputs(self):
        """Generate LOBO outputs."""
        # Save results to CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.output_dir / 'lobo_results.csv', index=False)

        # Save JSON
        with open(self.output_dir / 'lobo_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        # Generate LaTeX table
        self.generate_latex_table(results_df)

        self.log(f"\nResults saved to: {self.output_dir}")

    def generate_latex_table(self, results_df: pd.DataFrame):
        """Generate LaTeX table."""
        table_file = self.output_dir / 'lobo_table.tex'

        with open(table_file, 'w') as f:
            f.write(r"\begin{table}[t]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{LOBO Generalization Results}" + "\n")
            f.write(r"\label{tab:lobo}" + "\n")
            f.write(r"\begin{tabular}{lccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Category & Benchmarks & Train Makespan & Test Makespan \\" + "\n")
            f.write(r"\midrule" + "\n")

            # Group by category
            if 'category' in results_df.columns:
                for category in results_df['category'].unique():
                    cat_data = results_df[results_df['category'] == category]
                    n = len(cat_data)
                    f.write(f"{category} & {n} & -- & -- \\\\\n")

            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")

        self.log(f"LaTeX table saved to: {table_file}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='LOBO Training Server'
    )
    parser.add_argument(
        '--benchmarks', type=str, nargs='+',
        default=['all'],
        help='Benchmarks: "all", "bots", "polybench", or specific names'
    )
    parser.add_argument(
        '--method', type=str,
        default='MAMBRL_D3QN',
        choices=['MAMBRL_D3QN', 'MAMFRL_D3QN', 'SAMFRL', 'SAMBRL']
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output', type=str, default=str(save_dir))
    args = parser.parse_args()

    # Select benchmarks
    if 'all' in args.benchmarks:
        benchmarks = ALL_BENCHMARKS
    elif 'bots' in args.benchmarks:
        benchmarks = BOTS_BENCHMARKS
    elif 'polybench' in args.benchmarks:
        benchmarks = POLYBENCH_BENCHMARKS
    else:
        benchmarks = [b for b in ALL_BENCHMARKS if b['benchmark'] in args.benchmarks]

    if not benchmarks:
        logging.error("No benchmarks selected!")
        sys.exit(1)

    print("="*70)
    print("LOBO Training Server")
    print("="*70)
    print(f"Port: {PORT}")
    print(f"Method: {args.method}")
    print(f"Benchmarks: {len(benchmarks)}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print("="*70)

    # Establish connection
    client_socket, server_socket = establish_connection()

    try:
        experiment = LOBOExperiment(
            client_socket=client_socket,
            benchmarks=benchmarks,
            method=args.method,
            output_dir=Path(args.output)
        )

        results = experiment.run_full_lobo(
            seeds=args.seeds,
            epochs=args.epochs
        )

        print("\n" + "="*70)
        print(f"LOBO EXPERIMENT COMPLETE")
        print(f"Total evaluations: {len(results)}")
        print(f"Results saved to: {args.output}")
        print("="*70)

    finally:
        client_socket.close()
        server_socket.close()
        logging.info("Server shutdown")


if __name__ == '__main__':
    main()
