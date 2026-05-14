#!/usr/bin/env python3
"""
server_graphperf_experiment.py - GraphPerf-RT World Model Comparison Server

Compares:
1. MAMBRL with GraphPerf-RT GAT world model
2. MAMBRL with FCN environment model
3. MAMFRL (model-free baseline)

Based on zero_combined_twobench.py pattern from ZeroDVFS.

Server-Client Architecture:
- Server (this script): Runs on workstation, manages RL training
- Client (client.py): Runs on Jetson TX2, executes benchmarks

Usage:
    # On Jetson TX2 (client):
    python client.py

    # On Server (workstation):
    python server_graphperf_experiment.py --epochs 200 --seeds 42 123 456
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

# Default hyperparameters
DEFAULT_CONFIG = {
    'experiment_time': 200,
    'target_temp': 50,
    'clock_change_time': 30,
    'beta': 1.0,
    'learn_count': 32,
    'mem_size': 100000,
    'learning_rate': 0.1,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.9,
    'epsilon_min': 0.1,
    'epsilon_start': 1.0,
    'epsilon_end': 0.0,
    'batch_size': 32,
    'agent_train_start': 32,
    'plan_count': 20,
    'delayed_model_start': 30,
    'env_model_train_epochs': 20,
    'future_reward_weight': 1.0,
    'confidence_floor': 0.5,
    'reset_learning_rate_value': 20,
    'save_repetition': 20,
}

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

# Benchmarks
BOTS_APPLICATIONS = [
    {"benchmark": "fft", "variant": "bin-omp-tasks", "input_arg": "262144"},
    {"benchmark": "fft", "variant": "bin-omp-tasks-tied", "input_arg": "262144"},
    {"benchmark": "fft", "variant": "bin-serial", "input_arg": "262144"},
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

script_dir = Path(__file__).parent.absolute()
save_dir = script_dir / "save_model"
save_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS (from zero_combined_twobench.py)
# =============================================================================
def cleanup_memory():
    """Comprehensive memory cleanup between training modules."""
    plt.close('all')
    if TF_AVAILABLE:
        try:
            K.clear_session()
            if hasattr(tf, 'reset_default_graph'):
                tf.reset_default_graph()
        except Exception as e:
            logging.debug(f"TensorFlow cleanup warning: {e}")
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logging.debug(f"PyTorch cleanup warning: {e}")
    gc.collect()
    gc.collect()
    gc.collect()
    logging.info("Memory cleanup completed")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
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
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except:
            pass
    logging.info(f"Set random seed to {seed}")


def establish_connection() -> Tuple[socket.socket, socket.socket]:
    """Establish server connection and wait for client."""
    logging.info("Waiting for client connection...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((IP_ADDRESS, PORT))
    server_socket.listen(1)
    client_socket, address = server_socket.accept()
    logging.info(f"Client connected from {address[0]}:{address[1]}")
    return client_socket, server_socket


def parse_profiling_data(data: str) -> Optional[List[Dict[str, Any]]]:
    """Parse profiling data JSON from client."""
    try:
        msg = json.loads(data)
        profiling_data_list = msg.get('profiling_data_list', [])
        return profiling_data_list
    except json.JSONDecodeError as e:
        logging.warning(f"JSON decode error: {e}")
        return None


def profile_applications(
    client_socket: socket.socket,
    applications: List[Dict]
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Profile each application to get baseline performance data.

    Returns:
        application_profiles: Dict mapping (app_id, args) to profiling data
        profiling_data_list: List of all profiling data
        applications_fixed: List of application configs with IDs
    """
    application_profiles = {}
    profiling_data_list = []
    applications_fixed = []
    app_counter = 1

    for app_cfg in applications:
        application = {
            'id': app_counter,
            'benchmark': app_cfg['benchmark'],
            'variant': app_cfg['variant'],
            'input_arg': app_cfg['input_arg'],
            'app_args': app_cfg['input_arg'],
            'cores': '',
            'frequencies': [],
            'priority': None,
            'action': 'profile'
        }

        # Send application to client
        send_msg = json.dumps({'applications': [application]})
        client_socket.send((send_msg + '\n').encode())
        logging.info(f"Sent application {app_counter} ({app_cfg['benchmark']}) for profiling")

        # Wait for response
        data_received = False
        recv_buffer = ''
        start_time = time.time()
        timeout = 600

        while not data_received and time.time() - start_time < timeout:
            data = client_socket.recv(4096)
            if data:
                recv_buffer += data.decode()
                while '\n' in recv_buffer:
                    msg, recv_buffer = recv_buffer.split('\n', 1)
                    new_pd_list = parse_profiling_data(msg)
                    if new_pd_list is None:
                        continue
                    if len(new_pd_list) == 1:
                        data_received = True
                        profiling_data = new_pd_list[0]
                        app_id = profiling_data.get('application_id', application['id'])
                        app_args = profiling_data.get('app_args', application['app_args'])

                        # Calculate parallelism level
                        mk_one = profiling_data.get('makespan_one_core_frequency_11')
                        mk_all = profiling_data.get('makespan_all_cores_frequency_11')
                        if mk_one and mk_all and mk_all != 0:
                            parallelism = mk_one / mk_all
                        else:
                            parallelism = 1.0
                        profiling_data['parallelism_level'] = parallelism

                        application_profiles[(app_id, str(app_args))] = profiling_data
                        profiling_data_list.append(profiling_data)

                        # Send acknowledgment
                        ack = json.dumps({'status': 'received'})
                        client_socket.send((ack + '\n').encode())
                        logging.info(f"Received profiling data for app {app_id}")
            else:
                time.sleep(1)

        if not data_received:
            logging.error(f"Timeout waiting for profiling data for app {app_counter}")

        applications_fixed.append(application)
        app_counter += 1

    return application_profiles, profiling_data_list, applications_fixed


# =============================================================================
# THERMAL TRACKING (integrated from thermal_tracking_experiment.py)
# =============================================================================
class ThermalTracker:
    """Track thermal violations during training."""

    THRESHOLDS = {'tx2': 85.0, 'orin_nx': 85.0, 'rubikpi': 80.0}

    def __init__(self, method: str, platform: str = 'tx2'):
        self.method = method
        self.threshold = self.THRESHOLDS.get(platform, 85.0)
        self.violations = []
        self.peak_temps = []
        self.episode = 0

    def record(self, profiling_data: Dict):
        """Record thermal data from an episode."""
        self.episode += 1

        # Extract max temperature
        temps = []
        for key in profiling_data:
            if key.startswith('thermal_zone') and not key.endswith('_type'):
                val = profiling_data.get(key)
                if val and val > 0:
                    temps.append(val)

        if not temps and 'avg_temp_after' in profiling_data:
            temps = [profiling_data['avg_temp_after']]

        if temps:
            max_temp = max(temps)
            self.peak_temps.append(max_temp)
            if max_temp > self.threshold:
                self.violations.append({
                    'episode': self.episode,
                    'temp': max_temp
                })

    def get_summary(self) -> Dict:
        """Get thermal summary."""
        n = max(self.episode, 1)
        return {
            'method': self.method,
            'violations': len(self.violations),
            'violation_rate': len(self.violations) / n * 100,
            'peak_mean': np.mean(self.peak_temps) if self.peak_temps else 0,
            'peak_std': np.std(self.peak_temps) if self.peak_temps else 0,
            'peak_max': max(self.peak_temps) if self.peak_temps else 0,
        }


# =============================================================================
# WORLD MODEL COMPARISON EXPERIMENT
# =============================================================================
class WorldModelExperiment:
    """
    Compare GraphPerf-RT world model vs FCN environment model.

    Runs:
    1. MAMFRL (model-free baseline)
    2. MAMBRL with FCN environment model
    3. MAMBRL with GraphPerf-RT world model
    """

    def __init__(
        self,
        client_socket: socket.socket,
        application_profiles: Dict,
        profiling_data_list: List,
        applications_fixed: List,
        graphperf_model_path: Optional[str] = None,
        graphperf_config_path: Optional[str] = None,
        output_dir: Path = save_dir
    ):
        self.client_socket = client_socket
        self.application_profiles = application_profiles
        self.profiling_data_list = profiling_data_list
        self.applications_fixed = applications_fixed
        self.graphperf_model_path = graphperf_model_path
        self.graphperf_config_path = graphperf_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'MAMFRL': [],
            'MAMBRL_FCN': [],
            'MAMBRL_GraphPerf': [],
        }

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"experiment_log_{timestamp}.txt"

    def log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        logging.info(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def run_method(
        self,
        method_name: str,
        seed: int,
        epochs: int,
        config: Dict
    ) -> Dict:
        """
        Run a single method for comparison.

        In production, this calls the actual training function.
        """
        self.log(f"Running {method_name} (seed={seed}, epochs={epochs})")

        # Create thermal tracker
        thermal = ThermalTracker(method_name)

        # Generate output filename
        output_file = self.output_dir / f"server_{method_name.lower()}_{epochs}_seed{seed}.csv"

        # Import and run appropriate training function
        try:
            if method_name == 'MAMFRL':
                from server_mamfrl import train_fixed_mamfrl_d3qn
                # Run training...
                self.log(f"Would call train_fixed_mamfrl_d3qn")

            elif method_name == 'MAMBRL_FCN':
                from server_mambrl import train_fixed_mambrl_d3qn
                # Run with FCN environment model
                self.log(f"Would call train_fixed_mambrl_d3qn with FCN model")

            elif method_name == 'MAMBRL_GraphPerf':
                from server_mambrl import train_fixed_mambrl_d3qn
                # Run with GraphPerf-RT world model
                self.log(f"Would call train_fixed_mambrl_d3qn with GraphPerf-RT model")

        except ImportError as e:
            self.log(f"Import error: {e}")

        # Placeholder result
        result = {
            'method': method_name,
            'seed': seed,
            'epochs': epochs,
            'convergence_epoch': None,
            'final_makespan': None,
            'thermal': thermal.get_summary(),
            'output_file': str(output_file),
        }

        return result

    def run_comparison(
        self,
        seeds: List[int] = [42, 123, 456],
        epochs: int = 200
    ) -> Dict:
        """Run full comparison experiment."""
        self.log("="*70)
        self.log("WORLD MODEL COMPARISON EXPERIMENT")
        self.log(f"Methods: MAMFRL, MAMBRL_FCN, MAMBRL_GraphPerf")
        self.log(f"Seeds: {seeds}")
        self.log(f"Epochs: {epochs}")
        self.log("="*70)

        for seed in seeds:
            self.log(f"\n{'='*60}")
            self.log(f"SEED: {seed}")
            self.log(f"{'='*60}")

            set_seed(seed)

            for method in ['MAMFRL', 'MAMBRL_FCN', 'MAMBRL_GraphPerf']:
                self.log(f"\n--- {method} (seed={seed}) ---")

                result = self.run_method(
                    method_name=method,
                    seed=seed,
                    epochs=epochs,
                    config=DEFAULT_CONFIG
                )

                self.results[method].append(result)
                cleanup_memory()

        # Generate comparison outputs
        self.generate_outputs()

        return self.results

    def generate_outputs(self):
        """Generate comparison tables and plots."""
        # Save JSON results
        results_file = self.output_dir / 'world_model_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.log(f"Results saved to: {results_file}")

        # Generate LaTeX table
        self.generate_latex_table()

    def generate_latex_table(self):
        """Generate LaTeX comparison table."""
        table_file = self.output_dir / 'world_model_table.tex'

        with open(table_file, 'w') as f:
            f.write(r"\begin{table}[t]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{World Model Comparison: GraphPerf-RT vs FCN}" + "\n")
            f.write(r"\label{tab:world_model}" + "\n")
            f.write(r"\begin{tabular}{lcccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Method & Conv. Epoch & Makespan (s) & Thermal Viol. (\%) \\" + "\n")
            f.write(r"\midrule" + "\n")

            for method, runs in self.results.items():
                if runs:
                    # Aggregate statistics
                    f.write(f"{method} & -- & -- & -- \\\\\n")

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
        description='GraphPerf-RT World Model Comparison Server'
    )
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--graphperf-model', type=str, default=None)
    parser.add_argument('--graphperf-config', type=str, default=None)
    parser.add_argument('--output', type=str, default=str(save_dir))
    args = parser.parse_args()

    print("="*70)
    print("GraphPerf-RT World Model Comparison Server")
    print("="*70)
    print(f"Port: {PORT}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print("="*70)

    # Establish connection
    client_socket, server_socket = establish_connection()

    try:
        # Profile applications
        print("\n--- Profiling Applications ---")
        app_profiles, prof_data, apps_fixed = profile_applications(
            client_socket, BOTS_APPLICATIONS
        )

        # Run experiment
        experiment = WorldModelExperiment(
            client_socket=client_socket,
            application_profiles=app_profiles,
            profiling_data_list=prof_data,
            applications_fixed=apps_fixed,
            graphperf_model_path=args.graphperf_model,
            graphperf_config_path=args.graphperf_config,
            output_dir=Path(args.output)
        )

        results = experiment.run_comparison(
            seeds=args.seeds,
            epochs=args.epochs
        )

        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print(f"Results saved to: {args.output}")
        print("="*70)

    finally:
        client_socket.close()
        server_socket.close()
        logging.info("Server shutdown")


if __name__ == '__main__':
    main()
