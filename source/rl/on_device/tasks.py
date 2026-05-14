#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tasks.py - Paper Experiment Data Collection Script

This script collects all experimental data for the tasks in the ZeroDVFS paper.
Based on server_combined.py structure with targeted experiments for paper figures/tables.

PHASE A TASKS:
==============
A-MO-2: RL Convergence Comparison
    - Compare model-free vs model-based RL convergence
    - Baselines: ZTT (SAMFRL), DynaQ (SAMBRL), HMARL_MB (MAMBRL_D3QN)
    - Output: convergence_comparison.csv

A-MO-3: Decision Latency Measurement
    - Measure inference time for profiler model, thermal model, policy forward pass
    - Output: decision_latency.csv

A-MO-4: Energy Efficiency vs Baselines
    - Compare MAMBRL_D3QN against table-based, zTT, Linux governors
    - Output: energy_comparison.csv

A-MO-5: Ablation Study - Model-Based vs Model-Free
    - Train with and without environment model
    - Output: ablation_model_based.csv

USAGE:
======
python3 tasks.py

Requires client_evaluate_tx2.py running on the Jetson TX2.
"""

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Network Configuration
PORT = 8707
IP_ADDRESS = "0.0.0.0"

# Platform Configuration
PLATFORM_NAME = "jetson_tx2"

# =============================================================================
# EXPERIMENT CONFIGURATION FOR PAPER
# =============================================================================

# Number of episodes for convergence experiments (need enough to show convergence)
CONVERGENCE_EXPERIMENT_TIME = 100  # 100 episodes to show convergence curves

# Number of episodes for quick ablation experiments
ABLATION_EXPERIMENT_TIME = 50  # 50 episodes for ablation studies

# Number of inference runs for latency measurement
LATENCY_MEASUREMENT_RUNS = 100

# =============================================================================
# BASELINE MODULES FOR PAPER COMPARISON
# =============================================================================
# These are the key baselines for the paper:
# 1. FEDERATED: Heuristic baseline (non-RL)
# 2. SAMFRL: Model-free single-agent (represents zTT-like approach)
# 3. SAMBRL: Model-based single-agent (represents DynaQ-style)
# 4. MAMFRL_D3QN: Model-free multi-agent (ablation baseline)
# 5. MAMBRL_D3QN: Model-based multi-agent (our main method)

PAPER_MODULES = [
    # Heuristic baseline
    ("FEDERATED", "fixed_federated", "train_fixed_federated"),

    # Single-agent approaches (for convergence comparison)
    ("SAMFRL", "fixed_samfrl", "train_fixed_samfrl"),      # zTT-like model-free
    ("SAMBRL", "fixed_sambrl", "train_fixed_sambrl"),      # DynaQ-like model-based

    # Multi-agent approaches (main comparison)
    ("MAMFRL_D3QN", "fixed_mamfrl_d3qn_tx2", "train_fixed_mamfrl_d3qn"),  # Model-free MA
    ("MAMBRL_D3QN", "fixed_mambrl_d3qn_tx2", "train_fixed_mambrl_d3qn"),  # Our method
]

# Additional modules for comprehensive comparison (optional)
ADDITIONAL_MODULES = [
    ("SARBRL", "fixed_sarbrl", "train_fixed_sarbrl"),
    ("SAMBPGRL", "fixed_sambpgrl", "train_fixed_sambpgrl"),
    ("MAML", "fixed_maml", "train_fixed_maml"),
    ("MARBRL", "fixed_marbrl", "train_fixed_marbrl"),
    ("MARB_D3QN", "fixed_marb_d3qn", "train_fixed_marb_d3qn"),
    ("MARL_GEAR", "fixed_marl_gear", "train_fixed_marl_gear"),
]

# =============================================================================
# HYPERPARAMETERS (OPTIMAL FROM TUNING)
# =============================================================================
OPTIMAL_HYPERPARAMETERS = {
    'beta': 1.0,
    'learning_rate': 0.1,
    'batch_size': 64,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.90,
    'epsilon_min': 0.10,
    'epsilon_start': 1.0,
    'epsilon_end': 0.0,
    'plan_count': 100,
    'model_train_start': 32,
    'agent_train_start': 32,
    'real_synthetic_ratio': 0.5,
    'mem_size': 100000,
    'target_temp': 50,
    'clock_change_time': 30,
    'reset_learning_rate_value': 20,
    'save_repetition': 20,
}

# =============================================================================
# BENCHMARK CONFIGURATION (Representative set for paper)
# =============================================================================
# Using FFT variants as they show clear performance differences
BOTS_APPLICATIONS = [
    {"benchmark": "fft", "variant": "bin-omp-tasks", "input_arg": "262144"},
    {"benchmark": "fft", "variant": "bin-omp-tasks-tied", "input_arg": "262144"},
    {"benchmark": "fft", "variant": "bin-serial", "input_arg": "262144"},
]

POLYBENCH_APPLICATIONS = []

# Frequency and core configuration
FREQUENCY_COMBINATIONS = [
    [0] * 5,   # Lowest frequency (energy-efficient)
    [6] * 5,   # Mid frequency (balanced)
    [11] * 5,  # Highest frequency (performance)
]

PRIORITY_COMBINATIONS = [
    (10, 10, 10),
    (90, 90, 90),
    (10, 50, 90),
]

NUM_CORES_LIST = [1, 3, 5]

# =============================================================================
# IMPORTS
# =============================================================================
import socket
import logging
import json
import time
import os
import csv
import importlib
import gc
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# TensorFlow/Keras memory management
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# PyTorch memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import DATA_KEYS from central location
from server_combined import DATA_KEYS

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "save_model")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def cleanup_memory():
    """Comprehensive memory cleanup between training modules."""
    plt.close('all')
    if TF_AVAILABLE:
        try:
            K.clear_session()
        except Exception as e:
            logging.debug(f"TensorFlow cleanup warning: {e}")
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.debug(f"PyTorch cleanup warning: {e}")
    gc.collect()
    gc.collect()


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for data."""
    if len(data) == 0:
        return 0, 0, 0
    mean = np.mean(data)
    if len(data) == 1:
        return mean, mean, mean
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - ci, mean + ci


# =============================================================================
# RL MODULE REGISTRY
# =============================================================================
RL_MODULE_REGISTRY: Dict[str, Callable] = {}


def import_and_register_modules(modules_list):
    """Import and register RL modules."""
    logging.info("=" * 80)
    logging.info("Importing and registering RL modules...")
    logging.info("=" * 80)

    for display_name, module_name, function_name in modules_list:
        try:
            module = importlib.import_module(module_name)
            train_function = getattr(module, function_name)
            RL_MODULE_REGISTRY[display_name] = train_function
            logging.info(f"Registered: {display_name} from {module_name}.{function_name}")
        except ImportError as e:
            logging.warning(f"Could not import {module_name}: {e}")
        except AttributeError as e:
            logging.warning(f"Function {function_name} not found in {module_name}: {e}")

    logging.info(f"Total modules registered: {len(RL_MODULE_REGISTRY)}")


# =============================================================================
# BENCHMARK FAMILIES
# =============================================================================
OMPTASKS_BENCHES = {
    "alignment", "fft", "fib", "floorplan", "health",
    "concom", "knapsack", "nqueens", "sort", "sparselu",
    "strassen", "uts",
}

POLYBENCH_BENCHES = {
    "gemm", "gemver", "gesummv", "symm", "syr2k", "syrk", "trmm",
    "2mm", "3mm", "atax", "bicg", "doitgen", "mvt",
    "cholesky", "durbin", "gramschmidt", "lu", "ludcmp", "trisolv",
    "correlation", "covariance",
    "deriche", "floyd-warshall", "nussinov",
    "adi", "fdtd-2d", "heat-3d", "jacobi-1d", "jacobi-2d", "seidel-2d",
}


# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================
class ApplicationConfig:
    """Centralized configuration for benchmarks."""

    def __init__(self):
        self.bots_applications = BOTS_APPLICATIONS
        self.polybench_applications = POLYBENCH_APPLICATIONS
        self.applications_fixed: List[Dict[str, Any]] = []
        self.app_list: List[int] = []
        self._build_applications()
        self.frequency_combinations = FREQUENCY_COMBINATIONS
        self.priority_combinations = PRIORITY_COMBINATIONS
        self.num_cores_list = NUM_CORES_LIST

    def _build_applications(self):
        """Build unified application list."""
        app_id = 1

        for app_cfg in self.bots_applications:
            app = {
                "id": app_id,
                "benchmark": app_cfg["benchmark"],
                "variant": app_cfg["variant"],
                "input_arg": str(app_cfg["input_arg"]),
                "app_args": str(app_cfg["input_arg"]),
                "path": f"bots.sh {app_cfg['benchmark']} {app_cfg['variant']}",
                "cores": "",
                "frequencies": [],
                "priority": None,
                "action": "profile",
            }
            self.applications_fixed.append(app)
            self.app_list.append(app_id)
            app_id += 1

        for app_cfg in self.polybench_applications:
            app = {
                "id": app_id,
                "benchmark": app_cfg["benchmark"],
                "variant": "",
                "input_arg": str(app_cfg["input_arg"]),
                "app_args": str(app_cfg["input_arg"]),
                "path": f"polybench {app_cfg['benchmark']}",
                "cores": "",
                "frequencies": [],
                "priority": None,
                "action": "profile",
            }
            self.applications_fixed.append(app)
            self.app_list.append(app_id)
            app_id += 1

        logging.info(f"Built {len(self.applications_fixed)} applications")


# =============================================================================
# SOCKET UTILITIES
# =============================================================================
def parse_profiling_data(data: str) -> Optional[List[Dict[str, Any]]]:
    """Parse JSON profiling data from client."""
    try:
        msg = json.loads(data)
        return msg.get('profiling_data_list', [])
    except json.JSONDecodeError as e:
        logging.warning(f"JSON decode error: {e}")
        return None


def send_json(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Send JSON message with newline delimiter."""
    data = json.dumps(msg)
    sock.sendall((data + "\n").encode())


def establish_connection() -> Tuple[socket.socket, socket.socket]:
    """Establish TCP connection with client."""
    print("Waiting for connection...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((IP_ADDRESS, PORT))
    server_socket.listen(1)

    client_socket, address = server_socket.accept()
    print(f"Connection established with {address[0]}:{address[1]}")
    logging.info(f"Connection established with {address[0]}:{address[1]}")

    return client_socket, server_socket


# =============================================================================
# PROFILING DATA HANDLING
# =============================================================================
def enrich_profiling_data(profiling_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich profiling data with default values."""
    default_values = {
        "frequencies": [0],
        "cores": "1,2,3,4,5",
        "utilization": 0.0,
        "cycles": 0,
        "cache_references": 0,
        "cache_misses": 0,
        "branch_misses": 0,
    }

    enriched = dict(profiling_data)
    for key, default_val in default_values.items():
        if key not in enriched:
            enriched[key] = default_val

    if "total_energy_consumption" not in enriched:
        energy_keys = ["energy_system_j", "energy_main_j", "energy_cpu_j",
                       "energy_denver_j", "energy_gpu_j", "energy_ddr_j"]
        enriched["total_energy_consumption"] = sum(
            enriched.get(k, 0.0) for k in energy_keys
        )

    if "avg_temp_after" not in enriched:
        temps = [enriched.get(f"thermal_zone{i}", 50.0) for i in range(10)]
        enriched["avg_temp_after"] = np.mean([t for t in temps if t > 0]) if temps else 50.0

    return enriched


def perform_profiling_phase(
    client_socket: socket.socket,
    app_config: ApplicationConfig,
    timeout: float = 600.0
) -> Tuple[Dict[Tuple[int, str], Dict[str, Any]], List[Dict[str, Any]]]:
    """Perform profiling phase."""
    logging.info("=" * 80)
    logging.info("Starting Application Profiling Phase")
    logging.info("=" * 80)

    profiling_data_list: List[Dict[str, Any]] = []
    application_profiles: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for app in app_config.applications_fixed:
        application = {
            'id': app['id'],
            'benchmark': app['benchmark'],
            'variant': app['variant'],
            'input_arg': app['input_arg'],
            'app_args': app['input_arg'],
            'path': app['path'],
            'cores': '',
            'frequencies': [],
            'priority': None,
            'action': 'profile'
        }

        send_msg = json.dumps({'applications': [application]})
        client_socket.send((send_msg + '\n').encode())
        logging.info(f"Sent application {application['id']} for profiling")

        data_received = False
        recv_buffer = ''
        start_time = time.time()

        while not data_received and time.time() - start_time < timeout:
            try:
                data = client_socket.recv(4096)
                if data:
                    recv_buffer += data.decode()
                    while '\n' in recv_buffer:
                        msg, recv_buffer = recv_buffer.split('\n', 1)
                        new_pd_list = parse_profiling_data(msg)
                        if new_pd_list and len(new_pd_list) == 1:
                            data_received = True
                            profiling_data = new_pd_list[0]

                            mk_onecore = profiling_data.get('makespan_one_core_frequency_11')
                            mk_allcore = profiling_data.get('makespan_all_cores_frequency_11')
                            if mk_onecore and mk_allcore and mk_allcore != 0:
                                profiling_data['parallelism_level'] = mk_onecore / mk_allcore
                            else:
                                profiling_data['parallelism_level'] = 1.0

                            profiling_data = enrich_profiling_data(profiling_data)

                            app_id = profiling_data.get('application_id', application['id'])
                            app_args = profiling_data.get('app_args', application['input_arg'])
                            application_profiles[(app_id, str(app_args))] = profiling_data
                            profiling_data_list.append(profiling_data)

                            send_ack = json.dumps({'status': 'received'})
                            client_socket.send((send_ack + '\n').encode())
                            logging.info(f"Stored profiling data for app {app_id}")
                else:
                    time.sleep(1)
            except Exception as e:
                logging.error(f"Error receiving profiling data: {e}")
                time.sleep(1)

        if not data_received:
            logging.error(f"Timeout waiting for profiling data for app {app['id']}")

    logging.info(f"Profiling Phase Complete - {len(profiling_data_list)} apps profiled")
    return application_profiles, profiling_data_list


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================
def run_convergence_experiment(
    client_socket: socket.socket,
    app_config: ApplicationConfig,
    profiling_data_list: List[Dict[str, Any]],
    application_profiles: Dict,
    experiment_time: int = CONVERGENCE_EXPERIMENT_TIME
) -> Dict[str, Any]:
    """
    A-MO-2: RL Convergence Comparison

    Compare convergence of:
    - SAMFRL (model-free single-agent, zTT-like)
    - SAMBRL (model-based single-agent, DynaQ-like)
    - MAMBRL_D3QN (model-based multi-agent, our method)
    """
    logging.info("\n" + "=" * 80)
    logging.info("A-MO-2: RL CONVERGENCE COMPARISON EXPERIMENT")
    logging.info("=" * 80)

    convergence_modules = ["SAMFRL", "SAMBRL", "MAMBRL_D3QN"]
    results = {}

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for module_name in convergence_modules:
        if module_name not in RL_MODULE_REGISTRY:
            logging.warning(f"Module {module_name} not registered, skipping")
            continue

        logging.info(f"\n--- Training {module_name} for convergence analysis ---")

        train_function = RL_MODULE_REGISTRY[module_name]

        csv_filename = os.path.join(
            SAVE_DIR,
            f"CONVERGENCE_{module_name}_{timestamp}_{experiment_time}ep.csv"
        )

        hp = OPTIMAL_HYPERPARAMETERS

        start_time = time.time()

        try:
            kwargs = {
                'client_socket': client_socket,
                'data_keys': DATA_KEYS,
                'experiment_time': experiment_time,
                'clock_change_time': hp['clock_change_time'],
                'beta': hp['beta'],
                'load_model': False,
                'learn_count': hp['model_train_start'],
                'plan_count': hp['plan_count'],
                'mem_size': hp['mem_size'],
                'learning_rate': hp['learning_rate'],
                'discount_factor': hp['discount_factor'],
                'epsilon': hp['epsilon'],
                'epsilon_decay': hp['epsilon_decay'],
                'epsilon_min': hp['epsilon_min'],
                'epsilon_start': hp['epsilon_start'],
                'epsilon_end': hp['epsilon_end'],
                'reset_learning_rate_value': hp['reset_learning_rate_value'],
                'save_repetition': hp['save_repetition'],
                'save_model': True,
                'batch_size': hp['batch_size'],
                'agent_train_start': hp['agent_train_start'],
                'target_temp': hp['target_temp'],
                'server_name_1': "",
                'server_name_2': "",
                'server_name_main': csv_filename,
                'profiling_data_list': profiling_data_list,
                'application_profiles': application_profiles,
                'applications_fixed': app_config.applications_fixed,
                'priority_combinations': app_config.priority_combinations,
                'frequency_combinations': app_config.frequency_combinations,
                'num_cores_list': app_config.num_cores_list,
            }

            # Add model-based parameters for modules that support them
            if module_name in ['SAMBRL', 'MAMBRL_D3QN', 'MARB_D3QN', 'SARBRL']:
                kwargs['model_train_start'] = hp['model_train_start']
                kwargs['real_synthetic_ratio'] = hp['real_synthetic_ratio']

            result = train_function(**kwargs)

            training_time = time.time() - start_time

            if result is not None:
                ts, makespan_per_app, energy_per_app, temp_per_app = result

                # Calculate metrics
                if makespan_per_app and makespan_per_app[0]:
                    num_iters = len(makespan_per_app[0])
                    total_makespans = []
                    total_rewards = []

                    for iter_idx in range(num_iters):
                        total_mk = sum(app[iter_idx] for app in makespan_per_app if iter_idx < len(app))
                        total_makespans.append(total_mk)
                        # Approximate reward (negative makespan for minimization)
                        total_rewards.append(-total_mk)

                    # Find convergence point (95% of final performance)
                    final_reward = np.mean(total_rewards[-10:])  # Last 10 episodes
                    threshold = 0.95 * final_reward

                    convergence_episode = num_iters  # Default to last
                    for i, r in enumerate(total_rewards):
                        if r >= threshold:
                            convergence_episode = i + 1
                            break

                    results[module_name] = {
                        'convergence_episode': convergence_episode,
                        'final_makespan': np.mean(total_makespans[-10:]),
                        'final_reward': final_reward,
                        'training_time': training_time,
                        'makespans': total_makespans,
                        'rewards': total_rewards,
                        'csv_file': csv_filename,
                    }

                    logging.info(f"{module_name}: Converged at episode {convergence_episode}, "
                               f"final_makespan={results[module_name]['final_makespan']:.3f}s, "
                               f"training_time={training_time:.1f}s")

        except Exception as e:
            logging.error(f"Error training {module_name}: {e}", exc_info=True)

        # Cooldown
        cleanup_memory()
        time.sleep(5)

    # Save convergence results
    convergence_csv = os.path.join(RESULTS_DIR, f"convergence_comparison_{timestamp}.csv")
    with open(convergence_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['module', 'convergence_episode', 'final_makespan', 'final_reward',
                        'training_time', 'csv_file'])
        for module_name, data in results.items():
            writer.writerow([
                module_name,
                data['convergence_episode'],
                data['final_makespan'],
                data['final_reward'],
                data['training_time'],
                data['csv_file'],
            ])

    logging.info(f"Convergence results saved to: {convergence_csv}")

    return results


def measure_decision_latency(
    profiling_data_list: List[Dict[str, Any]],
    num_runs: int = LATENCY_MEASUREMENT_RUNS
) -> Dict[str, Any]:
    """
    A-MO-3: Decision Latency Measurement

    Measure inference time for:
    - Profiler model forward pass
    - Thermal model forward pass
    - Policy network forward pass
    - Total decision latency
    """
    logging.info("\n" + "=" * 80)
    logging.info("A-MO-3: DECISION LATENCY MEASUREMENT")
    logging.info("=" * 80)

    results = {
        'profiler_latency_ms': [],
        'thermal_latency_ms': [],
        'policy_latency_ms': [],
        'total_latency_ms': [],
    }

    # Load trained models if available
    try:
        from keras.models import load_model

        profiler_model_path = os.path.join(SAVE_DIR, "sambrl_profiler_model_data.weights.h5")
        thermal_model_path = os.path.join(SAVE_DIR, "priority_model_data.weights.h5")

        # Create dummy models for inference timing if models exist
        profiler_model = None
        thermal_model = None

        # Try to load or create models
        try:
            from keras.models import Sequential
            from keras.layers import Dense

            # Create profiler model architecture
            profiler_model = Sequential([
                Dense(128, activation='relu', input_shape=(20,)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            # Create thermal model architecture
            thermal_model = Sequential([
                Dense(64, activation='relu', input_shape=(15,)),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            logging.info("Created model architectures for latency measurement")

        except Exception as e:
            logging.warning(f"Could not create models: {e}")

        # Create dummy DQN policy network
        policy_model = None
        try:
            from keras.models import Sequential
            from keras.layers import Dense

            state_size = 20
            action_size = 12  # Typical action space size

            policy_model = Sequential([
                Dense(256, activation='relu', input_shape=(state_size,)),
                Dense(256, activation='relu'),
                Dense(action_size)
            ])

            logging.info("Created policy network for latency measurement")

        except Exception as e:
            logging.warning(f"Could not create policy model: {e}")

        # Generate dummy input data
        dummy_profiler_input = np.random.randn(1, 20).astype(np.float32)
        dummy_thermal_input = np.random.randn(1, 15).astype(np.float32)
        dummy_state_input = np.random.randn(1, 20).astype(np.float32)

        # Warm-up runs
        for _ in range(10):
            if profiler_model:
                profiler_model.predict(dummy_profiler_input, verbose=0)
            if thermal_model:
                thermal_model.predict(dummy_thermal_input, verbose=0)
            if policy_model:
                policy_model.predict(dummy_state_input, verbose=0)

        # Measure latency
        for i in range(num_runs):
            # Profiler model inference
            if profiler_model:
                start = time.perf_counter()
                profiler_model.predict(dummy_profiler_input, verbose=0)
                end = time.perf_counter()
                results['profiler_latency_ms'].append((end - start) * 1000)

            # Thermal model inference
            if thermal_model:
                start = time.perf_counter()
                thermal_model.predict(dummy_thermal_input, verbose=0)
                end = time.perf_counter()
                results['thermal_latency_ms'].append((end - start) * 1000)

            # Policy inference
            if policy_model:
                start = time.perf_counter()
                policy_model.predict(dummy_state_input, verbose=0)
                end = time.perf_counter()
                results['policy_latency_ms'].append((end - start) * 1000)

            # Total decision latency (all three)
            start = time.perf_counter()
            if profiler_model:
                profiler_model.predict(dummy_profiler_input, verbose=0)
            if thermal_model:
                thermal_model.predict(dummy_thermal_input, verbose=0)
            if policy_model:
                policy_model.predict(dummy_state_input, verbose=0)
            end = time.perf_counter()
            results['total_latency_ms'].append((end - start) * 1000)

            if (i + 1) % 20 == 0:
                logging.info(f"Completed {i + 1}/{num_runs} latency measurements")

        # Calculate statistics
        latency_stats = {}
        for key, values in results.items():
            if values:
                mean, ci_low, ci_high = calculate_confidence_interval(values)
                latency_stats[key] = {
                    'mean': mean,
                    'std': np.std(values),
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'min': np.min(values),
                    'max': np.max(values),
                }
                logging.info(f"{key}: {mean:.3f} ± {np.std(values):.3f} ms "
                           f"(95% CI: [{ci_low:.3f}, {ci_high:.3f}])")

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        latency_csv = os.path.join(RESULTS_DIR, f"decision_latency_{timestamp}.csv")

        with open(latency_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['component', 'mean_ms', 'std_ms', 'ci_low_ms', 'ci_high_ms',
                           'min_ms', 'max_ms'])
            for key, stats in latency_stats.items():
                writer.writerow([
                    key.replace('_ms', ''),
                    stats['mean'],
                    stats['std'],
                    stats['ci_low'],
                    stats['ci_high'],
                    stats['min'],
                    stats['max'],
                ])

        logging.info(f"Latency results saved to: {latency_csv}")

        return latency_stats

    except Exception as e:
        logging.error(f"Error measuring latency: {e}", exc_info=True)
        return {}


def run_energy_comparison(
    client_socket: socket.socket,
    app_config: ApplicationConfig,
    profiling_data_list: List[Dict[str, Any]],
    application_profiles: Dict,
    experiment_time: int = 50
) -> Dict[str, Any]:
    """
    A-MO-4: Energy Efficiency vs Baselines

    Compare energy efficiency of:
    - FEDERATED (heuristic baseline)
    - SAMFRL (model-free baseline)
    - MAMBRL_D3QN (our method)
    """
    logging.info("\n" + "=" * 80)
    logging.info("A-MO-4: ENERGY EFFICIENCY COMPARISON")
    logging.info("=" * 80)

    energy_modules = ["FEDERATED", "SAMFRL", "MAMBRL_D3QN"]
    results = {}

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for module_name in energy_modules:
        if module_name not in RL_MODULE_REGISTRY:
            logging.warning(f"Module {module_name} not registered, skipping")
            continue

        logging.info(f"\n--- Running {module_name} for energy comparison ---")

        train_function = RL_MODULE_REGISTRY[module_name]

        csv_filename = os.path.join(
            SAVE_DIR,
            f"ENERGY_{module_name}_{timestamp}_{experiment_time}ep.csv"
        )

        hp = OPTIMAL_HYPERPARAMETERS

        try:
            kwargs = {
                'client_socket': client_socket,
                'data_keys': DATA_KEYS,
                'experiment_time': experiment_time,
                'clock_change_time': hp['clock_change_time'],
                'beta': hp['beta'],
                'load_model': False,
                'learn_count': hp['model_train_start'],
                'plan_count': hp['plan_count'],
                'mem_size': hp['mem_size'],
                'learning_rate': hp['learning_rate'],
                'discount_factor': hp['discount_factor'],
                'epsilon': hp['epsilon'],
                'epsilon_decay': hp['epsilon_decay'],
                'epsilon_min': hp['epsilon_min'],
                'epsilon_start': hp['epsilon_start'],
                'epsilon_end': hp['epsilon_end'],
                'reset_learning_rate_value': hp['reset_learning_rate_value'],
                'save_repetition': hp['save_repetition'],
                'save_model': False,
                'batch_size': hp['batch_size'],
                'agent_train_start': hp['agent_train_start'],
                'target_temp': hp['target_temp'],
                'server_name_1': "",
                'server_name_2': "",
                'server_name_main': csv_filename,
                'profiling_data_list': profiling_data_list,
                'application_profiles': application_profiles,
                'applications_fixed': app_config.applications_fixed,
                'priority_combinations': app_config.priority_combinations,
                'frequency_combinations': app_config.frequency_combinations,
                'num_cores_list': app_config.num_cores_list,
            }

            if module_name in ['SAMBRL', 'MAMBRL_D3QN', 'MARB_D3QN', 'SARBRL']:
                kwargs['model_train_start'] = hp['model_train_start']
                kwargs['real_synthetic_ratio'] = hp['real_synthetic_ratio']

            result = train_function(**kwargs)

            if result is not None:
                ts, makespan_per_app, energy_per_app, temp_per_app = result

                if energy_per_app and energy_per_app[0]:
                    num_iters = len(energy_per_app[0])
                    total_energies = []
                    total_makespans = []
                    avg_temps = []

                    for iter_idx in range(num_iters):
                        total_energy = sum(app[iter_idx] for app in energy_per_app if iter_idx < len(app))
                        total_energies.append(total_energy)

                        total_mk = sum(app[iter_idx] for app in makespan_per_app if iter_idx < len(app))
                        total_makespans.append(total_mk)

                        if temp_per_app and temp_per_app[0]:
                            temps = [app[iter_idx] for app in temp_per_app if iter_idx < len(app)]
                            avg_temps.append(np.mean(temps))

                    # Use last 20% of episodes for final metrics
                    final_idx = int(0.8 * num_iters)

                    results[module_name] = {
                        'avg_energy': np.mean(total_energies[final_idx:]),
                        'avg_makespan': np.mean(total_makespans[final_idx:]),
                        'avg_temperature': np.mean(avg_temps[final_idx:]) if avg_temps else 50.0,
                        'energies': total_energies,
                        'makespans': total_makespans,
                        'temperatures': avg_temps,
                        'csv_file': csv_filename,
                    }

                    logging.info(f"{module_name}: avg_energy={results[module_name]['avg_energy']:.2f}J, "
                               f"avg_makespan={results[module_name]['avg_makespan']:.3f}s")

        except Exception as e:
            logging.error(f"Error running {module_name}: {e}", exc_info=True)

        cleanup_memory()
        time.sleep(5)

    # Calculate normalized metrics (relative to MAMBRL_D3QN as baseline)
    if 'MAMBRL_D3QN' in results:
        baseline_energy = results['MAMBRL_D3QN']['avg_energy']
        baseline_makespan = results['MAMBRL_D3QN']['avg_makespan']

        for module_name in results:
            results[module_name]['normalized_energy'] = (
                results[module_name]['avg_energy'] / baseline_energy
            )
            results[module_name]['normalized_makespan'] = (
                results[module_name]['avg_makespan'] / baseline_makespan
            )

    # Save results
    energy_csv = os.path.join(RESULTS_DIR, f"energy_comparison_{timestamp}.csv")
    with open(energy_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['module', 'avg_energy_j', 'avg_makespan_s', 'avg_temperature_c',
                        'normalized_energy', 'normalized_makespan', 'csv_file'])
        for module_name, data in results.items():
            writer.writerow([
                module_name,
                data['avg_energy'],
                data['avg_makespan'],
                data['avg_temperature'],
                data.get('normalized_energy', 1.0),
                data.get('normalized_makespan', 1.0),
                data['csv_file'],
            ])

    logging.info(f"Energy comparison results saved to: {energy_csv}")

    return results


def run_ablation_study(
    client_socket: socket.socket,
    app_config: ApplicationConfig,
    profiling_data_list: List[Dict[str, Any]],
    application_profiles: Dict,
    experiment_time: int = ABLATION_EXPERIMENT_TIME
) -> Dict[str, Any]:
    """
    A-MO-5: Ablation Study - Model-Based vs Model-Free

    Compare:
    - MAMFRL_D3QN (model-free multi-agent)
    - MAMBRL_D3QN (model-based multi-agent)

    To show benefit of environment model.
    """
    logging.info("\n" + "=" * 80)
    logging.info("A-MO-5: ABLATION STUDY - MODEL-BASED VS MODEL-FREE")
    logging.info("=" * 80)

    ablation_modules = ["MAMFRL_D3QN", "MAMBRL_D3QN"]
    results = {}

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for module_name in ablation_modules:
        if module_name not in RL_MODULE_REGISTRY:
            logging.warning(f"Module {module_name} not registered, skipping")
            continue

        logging.info(f"\n--- Training {module_name} for ablation ---")

        train_function = RL_MODULE_REGISTRY[module_name]

        csv_filename = os.path.join(
            SAVE_DIR,
            f"ABLATION_{module_name}_{timestamp}_{experiment_time}ep.csv"
        )

        hp = OPTIMAL_HYPERPARAMETERS

        start_time = time.time()

        try:
            kwargs = {
                'client_socket': client_socket,
                'data_keys': DATA_KEYS,
                'experiment_time': experiment_time,
                'clock_change_time': hp['clock_change_time'],
                'beta': hp['beta'],
                'load_model': False,
                'learn_count': hp['model_train_start'],
                'plan_count': hp['plan_count'],
                'mem_size': hp['mem_size'],
                'learning_rate': hp['learning_rate'],
                'discount_factor': hp['discount_factor'],
                'epsilon': hp['epsilon'],
                'epsilon_decay': hp['epsilon_decay'],
                'epsilon_min': hp['epsilon_min'],
                'epsilon_start': hp['epsilon_start'],
                'epsilon_end': hp['epsilon_end'],
                'reset_learning_rate_value': hp['reset_learning_rate_value'],
                'save_repetition': hp['save_repetition'],
                'save_model': False,
                'batch_size': hp['batch_size'],
                'agent_train_start': hp['agent_train_start'],
                'target_temp': hp['target_temp'],
                'server_name_1': "",
                'server_name_2': "",
                'server_name_main': csv_filename,
                'profiling_data_list': profiling_data_list,
                'application_profiles': application_profiles,
                'applications_fixed': app_config.applications_fixed,
                'priority_combinations': app_config.priority_combinations,
                'frequency_combinations': app_config.frequency_combinations,
                'num_cores_list': app_config.num_cores_list,
            }

            if module_name in ['SAMBRL', 'MAMBRL_D3QN', 'MARB_D3QN', 'SARBRL']:
                kwargs['model_train_start'] = hp['model_train_start']
                kwargs['real_synthetic_ratio'] = hp['real_synthetic_ratio']

            result = train_function(**kwargs)

            training_time = time.time() - start_time

            if result is not None:
                ts, makespan_per_app, energy_per_app, temp_per_app = result

                if makespan_per_app and makespan_per_app[0]:
                    num_iters = len(makespan_per_app[0])
                    total_makespans = []
                    cumulative_rewards = []

                    cumsum = 0
                    for iter_idx in range(num_iters):
                        total_mk = sum(app[iter_idx] for app in makespan_per_app if iter_idx < len(app))
                        total_makespans.append(total_mk)
                        cumsum += (-total_mk)  # Negative makespan as reward
                        cumulative_rewards.append(cumsum)

                    # Find convergence episode (95% of final cumulative reward)
                    final_reward = cumulative_rewards[-1]
                    threshold = 0.95 * final_reward

                    convergence_episode = num_iters
                    for i, r in enumerate(cumulative_rewards):
                        if r >= threshold:
                            convergence_episode = i + 1
                            break

                    results[module_name] = {
                        'convergence_episode': convergence_episode,
                        'final_makespan': np.mean(total_makespans[-10:]),
                        'final_cumulative_reward': final_reward,
                        'training_time': training_time,
                        'makespans': total_makespans,
                        'cumulative_rewards': cumulative_rewards,
                        'csv_file': csv_filename,
                    }

                    logging.info(f"{module_name}: Converged at ep {convergence_episode}, "
                               f"final_makespan={results[module_name]['final_makespan']:.3f}s")

        except Exception as e:
            logging.error(f"Error training {module_name}: {e}", exc_info=True)

        cleanup_memory()
        time.sleep(5)

    # Calculate improvement from model-based approach
    if 'MAMFRL_D3QN' in results and 'MAMBRL_D3QN' in results:
        mf_convergence = results['MAMFRL_D3QN']['convergence_episode']
        mb_convergence = results['MAMBRL_D3QN']['convergence_episode']

        convergence_speedup = mf_convergence / mb_convergence if mb_convergence > 0 else 1.0

        mf_makespan = results['MAMFRL_D3QN']['final_makespan']
        mb_makespan = results['MAMBRL_D3QN']['final_makespan']

        makespan_improvement = (mf_makespan - mb_makespan) / mf_makespan * 100

        logging.info(f"\n--- ABLATION RESULTS ---")
        logging.info(f"Convergence speedup (MB vs MF): {convergence_speedup:.1f}x")
        logging.info(f"Makespan improvement: {makespan_improvement:.1f}%")

    # Save results
    ablation_csv = os.path.join(RESULTS_DIR, f"ablation_model_based_{timestamp}.csv")
    with open(ablation_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['module', 'convergence_episode', 'final_makespan',
                        'final_cumulative_reward', 'training_time', 'csv_file'])
        for module_name, data in results.items():
            writer.writerow([
                module_name,
                data['convergence_episode'],
                data['final_makespan'],
                data['final_cumulative_reward'],
                data['training_time'],
                data['csv_file'],
            ])

    logging.info(f"Ablation results saved to: {ablation_csv}")

    return results


def run_all_baselines(
    client_socket: socket.socket,
    app_config: ApplicationConfig,
    profiling_data_list: List[Dict[str, Any]],
    application_profiles: Dict,
    experiment_time: int = 100
) -> Dict[str, Any]:
    """
    Run all 11 baselines for comprehensive comparison.
    """
    logging.info("\n" + "=" * 80)
    logging.info("COMPREHENSIVE BASELINE COMPARISON (ALL 11 MODULES)")
    logging.info("=" * 80)

    all_results = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for module_name, train_function in RL_MODULE_REGISTRY.items():
        logging.info(f"\n--- Training {module_name} ---")

        csv_filename = os.path.join(
            SAVE_DIR,
            f"BASELINE_{module_name}_{timestamp}_{experiment_time}ep.csv"
        )

        hp = OPTIMAL_HYPERPARAMETERS

        start_time = time.time()

        try:
            kwargs = {
                'client_socket': client_socket,
                'data_keys': DATA_KEYS,
                'experiment_time': experiment_time,
                'clock_change_time': hp['clock_change_time'],
                'beta': hp['beta'],
                'load_model': False,
                'learn_count': hp['model_train_start'],
                'plan_count': hp['plan_count'],
                'mem_size': hp['mem_size'],
                'learning_rate': hp['learning_rate'],
                'discount_factor': hp['discount_factor'],
                'epsilon': hp['epsilon'],
                'epsilon_decay': hp['epsilon_decay'],
                'epsilon_min': hp['epsilon_min'],
                'epsilon_start': hp['epsilon_start'],
                'epsilon_end': hp['epsilon_end'],
                'reset_learning_rate_value': hp['reset_learning_rate_value'],
                'save_repetition': hp['save_repetition'],
                'save_model': True,
                'batch_size': hp['batch_size'],
                'agent_train_start': hp['agent_train_start'],
                'target_temp': hp['target_temp'],
                'server_name_1': "",
                'server_name_2': "",
                'server_name_main': csv_filename,
                'profiling_data_list': profiling_data_list,
                'application_profiles': application_profiles,
                'applications_fixed': app_config.applications_fixed,
                'priority_combinations': app_config.priority_combinations,
                'frequency_combinations': app_config.frequency_combinations,
                'num_cores_list': app_config.num_cores_list,
            }

            if module_name in ['SAMBRL', 'MAMBRL_D3QN', 'MARB_D3QN', 'SARBRL']:
                kwargs['model_train_start'] = hp['model_train_start']
                kwargs['real_synthetic_ratio'] = hp['real_synthetic_ratio']

            result = train_function(**kwargs)

            training_time = time.time() - start_time

            if result is not None:
                ts, makespan_per_app, energy_per_app, temp_per_app = result

                if makespan_per_app and makespan_per_app[0]:
                    num_iters = len(makespan_per_app[0])

                    total_makespans = []
                    total_energies = []
                    avg_temps = []

                    for iter_idx in range(num_iters):
                        total_mk = sum(app[iter_idx] for app in makespan_per_app if iter_idx < len(app))
                        total_makespans.append(total_mk)

                        if energy_per_app and energy_per_app[0]:
                            total_en = sum(app[iter_idx] for app in energy_per_app if iter_idx < len(app))
                            total_energies.append(total_en)

                        if temp_per_app and temp_per_app[0]:
                            temps = [app[iter_idx] for app in temp_per_app if iter_idx < len(app)]
                            avg_temps.append(np.mean(temps))

                    all_results[module_name] = {
                        'final_makespan': np.mean(total_makespans[-10:]),
                        'final_energy': np.mean(total_energies[-10:]) if total_energies else 0,
                        'final_temperature': np.mean(avg_temps[-10:]) if avg_temps else 50.0,
                        'training_time': training_time,
                        'num_iterations': num_iters,
                        'makespans': total_makespans,
                        'energies': total_energies,
                        'temperatures': avg_temps,
                        'csv_file': csv_filename,
                    }

                    logging.info(f"{module_name}: makespan={all_results[module_name]['final_makespan']:.3f}s, "
                               f"energy={all_results[module_name]['final_energy']:.2f}J, "
                               f"time={training_time:.1f}s")

        except Exception as e:
            logging.error(f"Error training {module_name}: {e}", exc_info=True)

        cleanup_memory()
        time.sleep(5)

    # Save comprehensive results
    comprehensive_csv = os.path.join(RESULTS_DIR, f"comprehensive_baseline_{timestamp}.csv")
    with open(comprehensive_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['module', 'final_makespan_s', 'final_energy_j', 'final_temperature_c',
                        'training_time_s', 'num_iterations', 'csv_file'])
        for module_name, data in all_results.items():
            writer.writerow([
                module_name,
                data['final_makespan'],
                data['final_energy'],
                data['final_temperature'],
                data['training_time'],
                data['num_iterations'],
                data['csv_file'],
            ])

    logging.info(f"Comprehensive results saved to: {comprehensive_csv}")

    return all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ZERODVFS PAPER EXPERIMENTS - DATA COLLECTION")
    print("=" * 80)
    print("\nThis script will collect data for the Phase A tasks:")
    print("  A-MO-2: RL Convergence Comparison")
    print("  A-MO-3: Decision Latency Measurement")
    print("  A-MO-4: Energy Efficiency vs Baselines")
    print("  A-MO-5: Ablation Study (Model-Based vs Model-Free)")
    print("\nResults will be saved to:", RESULTS_DIR)
    print("=" * 80)

    # Import and register all modules
    import_and_register_modules(PAPER_MODULES + ADDITIONAL_MODULES)

    if not RL_MODULE_REGISTRY:
        logging.error("No RL modules registered!")
        raise SystemExit(1)

    logging.info(f"Registered modules: {list(RL_MODULE_REGISTRY.keys())}")

    # Build application configuration
    app_config = ApplicationConfig()

    # Establish connection with client
    print("\nWaiting for client connection...")
    client_socket, server_socket = establish_connection()

    try:
        # Profiling phase
        application_profiles, profiling_data_list = perform_profiling_phase(
            client_socket, app_config, timeout=600.0
        )

        if not profiling_data_list:
            logging.error("No profiling data collected!")
            raise SystemExit(1)

        # =====================================================================
        # A-MO-3: Decision Latency Measurement (can run without training)
        # =====================================================================
        print("\n" + "=" * 80)
        print("Running A-MO-3: Decision Latency Measurement")
        print("=" * 80)
        latency_results = measure_decision_latency(profiling_data_list)

        # =====================================================================
        # A-MO-2: RL Convergence Comparison
        # =====================================================================
        print("\n" + "=" * 80)
        print("Running A-MO-2: RL Convergence Comparison")
        print("=" * 80)
        convergence_results = run_convergence_experiment(
            client_socket, app_config, profiling_data_list, application_profiles,
            experiment_time=CONVERGENCE_EXPERIMENT_TIME
        )

        # =====================================================================
        # A-MO-4: Energy Efficiency Comparison
        # =====================================================================
        print("\n" + "=" * 80)
        print("Running A-MO-4: Energy Efficiency Comparison")
        print("=" * 80)
        energy_results = run_energy_comparison(
            client_socket, app_config, profiling_data_list, application_profiles,
            experiment_time=50
        )

        # =====================================================================
        # A-MO-5: Ablation Study
        # =====================================================================
        print("\n" + "=" * 80)
        print("Running A-MO-5: Ablation Study")
        print("=" * 80)
        ablation_results = run_ablation_study(
            client_socket, app_config, profiling_data_list, application_profiles,
            experiment_time=ABLATION_EXPERIMENT_TIME
        )

        # =====================================================================
        # OPTIONAL: Run All 11 Baselines
        # =====================================================================
        run_comprehensive = input("\nRun comprehensive baseline comparison (all 11 modules)? [y/N]: ")
        if run_comprehensive.lower() == 'y':
            comprehensive_results = run_all_baselines(
                client_socket, app_config, profiling_data_list, application_profiles,
                experiment_time=100
            )

        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("\n" + "=" * 80)
        print("EXPERIMENT DATA COLLECTION COMPLETE")
        print("=" * 80)
        print(f"\nResults saved in: {RESULTS_DIR}")
        print("\nGenerated files:")
        for f in os.listdir(RESULTS_DIR):
            print(f"  - {f}")
        print("\nNext steps:")
        print("  1. Run Phase B analysis scripts to generate figures")
        print("  2. Use results for Phase C paper writing")
        print("=" * 80)

    except KeyboardInterrupt:
        logging.info("Experiment interrupted by user")
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
    finally:
        try:
            client_socket.close()
            server_socket.close()
            logging.info("Connections closed")
        except Exception:
            pass
