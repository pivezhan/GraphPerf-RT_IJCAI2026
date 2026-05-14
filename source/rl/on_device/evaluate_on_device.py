#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_rl_baselines_tx2.py - LIVE RL Baseline Evaluation for TX2

This script evaluates RL baselines on the Jetson TX2 platform using LIVE execution
with the client.py client. Uses the EXACT same communication protocol
and methodology as server_combined.py.

CONVERGENCE ANALYSIS:
=====================
Compares Model-Based (MB) vs Model-Free (MF) approaches:
- Single-Agent: SAMBRL (MB-SA) vs SAMFRL (MF-SA)
- Multi-Agent: MAMBRL_D3QN (MB-MA) vs MAMFRL_D3QN (MF-MA)

FEDERATED is NOT included in convergence plots (it's a heuristic, not learning-based).
FEDERATED is only shown in computational complexity comparison table.

COMPUTATIONAL COMPLEXITY TABLE:
===============================
FEDERATED: O(n_apps * n_speeds * n_core_configs) - requires profiling table collection
           for each app × frequency level × core configuration combination
SAMFRL/MAMFRL: O(N) - model-free requires N real environment samples for learning
SAMBRL/MAMBRL: O(N_real) + synthetic - uses world model to generate synthetic samples
GraphPerf-RT: Zero-shot via GAT prediction - latency depends only on model API calls

USAGE:
======
python3 evaluate_rl_baselines_tx2.py                    # Run all evaluations
python3 evaluate_rl_baselines_tx2.py --task convergence # Only convergence (SA/MA: MB vs MF)
python3 evaluate_rl_baselines_tx2.py --task complexity  # Only complexity table
python3 evaluate_rl_baselines_tx2.py --tune             # Enable HP tuning
"""

# =============================================================================
# IMPORTS
# =============================================================================
import socket
import logging
import json
import time
import os
import sys
import argparse
import importlib
import gc
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Add the script directory to sys.path so we can import sibling modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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

# =============================================================================
# CONFIGURATION
# =============================================================================
PORT = 8707
IP_ADDRESS = "0.0.0.0"
PLATFORM_NAME = "jetson_tx2"

HP_TUNING_EXPERIMENT_TIME = 50
MAIN_EXPERIMENT_TIME = 200
N_SEEDS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

ENABLE_HP_TUNING = False
HP_TUNING_MODULE = "SAMBRL"

HP_TUNE_CONFIG = {
    'plan_count': False, 'model_train_start': False, 'agent_train_start': False,
    'real_synthetic_ratio': False, 'learning_rate': False, 'batch_size': False,
    'beta': False, 'epsilon_min': False, 'epsilon_decay': False, 'discount_factor': False,
}

HP_SEARCH_VALUES = {
    'plan_count': [100, 200], 'model_train_start': [16, 32], 'agent_train_start': [16, 32],
    'real_synthetic_ratio': [0.5, 0.7], 'learning_rate': [0.1], 'batch_size': [64],
    'beta': [1.0], 'epsilon_min': [0.1], 'epsilon_decay': [0.90], 'discount_factor': [0.99],
}

DEFAULT_HYPERPARAMETERS = {
    'beta': 1.0, 'learning_rate': 0.1, 'batch_size': 32, 'discount_factor': 0.99,
    'epsilon': 1.0, 'epsilon_decay': 0.90, 'epsilon_min': 0.10, 'epsilon_start': 1.0,
    'epsilon_end': 0.0, 'plan_count': 100, 'model_train_start': 32, 'agent_train_start': 32,
    'real_synthetic_ratio': 0.5, 'mem_size': 100000, 'reset_learning_rate_value': 20,
    'save_repetition': 20, 'save_model': True, 'load_model': False, 'target_temp': 50,
    'clock_change_time': 30, 'learn_count': 16,
}

# =============================================================================
# PLATFORM CONFIGURATION (for complexity calculation)
# =============================================================================
# TX2 Configuration
TX2_NUM_SPEEDS = 12        # Frequency levels (0-11)
TX2_NUM_CORE_CONFIGS = 5   # Core configurations (1,2,3,4,5 cores)
TX2_TOTAL_CONFIGS = TX2_NUM_SPEEDS * TX2_NUM_CORE_CONFIGS  # 60 configs per app

# =============================================================================
# RL MODULES - Convergence modules (MB vs MF for SA and MA)
# =============================================================================
CONVERGENCE_MODULES = [
    ("SAMFRL", "fixed_samfrl", "train_fixed_samfrl"),           # Model-Free Single-Agent
    ("SAMBRL", "fixed_sambrl", "train_fixed_sambrl"),           # Model-Based Single-Agent
    ("MAMFRL_D3QN", "fixed_mamfrl_d3qn_tx2", "train_fixed_mamfrl_d3qn"),  # Model-Free Multi-Agent
    ("MAMBRL_D3QN", "fixed_mambrl_d3qn_tx2", "train_fixed_mambrl_d3qn"),  # Model-Based Multi-Agent
]

ALL_MODULES = [
    ("FEDERATED", "fixed_federated", "train_fixed_federated"),
] + CONVERGENCE_MODULES

BOTS_APPLICATIONS = [
    {"benchmark": "fft", "variant": "bin-omp-tasks", "input_arg": "262144"},
    {"benchmark": "fft", "variant": "bin-omp-tasks-tied", "input_arg": "262144"},
    {"benchmark": "fft", "variant": "bin-serial", "input_arg": "262144"},
]
POLYBENCH_APPLICATIONS = []

FREQUENCY_COMBINATIONS = [[0]*5, [2]*5, [6]*5, [11]*5]
PRIORITY_COMBINATIONS = [(10,10,10), (90,90,90), (10,50,90), (50,90,10), (90,10,50), (10,90,50)]
NUM_CORES_LIST = [1, 3, 5]

# =============================================================================
# PATHS
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# SCRIPT_DIR already defined at top of file for sys.path
SAVE_DIR = os.path.join(SCRIPT_DIR, "save_model")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "paper_results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

TUNED_HP_FILE = os.path.join(SAVE_DIR, f"tuned_hyperparameters_{PLATFORM_NAME}.json")

DATA_KEYS = ["makespan", "num_cores", "cores", "qmax", "energy", "freq",
             "priority_combination", "thermal", "branchmisses", "cachemisses", "priority", "reward"]

MODULES_WITH_MODEL_TRAIN_START = ['SAMBRL', 'MAMBRL_D3QN', 'MARB_D3QN', 'SARBRL']

# =============================================================================
# RL MODULE DISPLAY CONFIG (for plots)
# =============================================================================
RL_MODULES = {
    'FEDERATED': {'display_name': 'FEDERATED (Heuristic)', 'type': 'heuristic', 'color': '#7f7f7f', 'marker': 'x', 'linestyle': ':'},
    'SAMFRL': {'display_name': 'SAMFRL (MF-SA)', 'type': 'model-free', 'agent_type': 'single', 'color': '#1f77b4', 'marker': 'o', 'linestyle': '--'},
    'SAMBRL': {'display_name': 'SAMBRL (MB-SA)', 'type': 'model-based', 'agent_type': 'single', 'color': '#2ca02c', 'marker': 's', 'linestyle': '-'},
    'MAMFRL_D3QN': {'display_name': 'MAMFRL (MF-MA)', 'type': 'model-free', 'agent_type': 'multi', 'color': '#ff7f0e', 'marker': '^', 'linestyle': '--'},
    'MAMBRL_D3QN': {'display_name': 'MAMBRL (MB-MA)', 'type': 'model-based', 'agent_type': 'multi', 'color': '#d62728', 'marker': 'D', 'linestyle': '-'},
    'GraphPerf-RT': {'display_name': 'GraphPerf-RT (Ours)', 'type': 'zero-shot', 'agent_type': 'single', 'color': '#9467bd', 'marker': '*', 'linestyle': '-'},
}

# =============================================================================
# MEMORY CLEANUP
# =============================================================================
def cleanup_memory():
    plt.close('all')
    if TF_AVAILABLE:
        try:
            K.clear_session()
        except Exception:
            pass
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    logging.info("Memory cleanup completed")

# =============================================================================
# HYPERPARAMETER SAVE/LOAD
# =============================================================================
def save_tuned_hyperparameters(hp_dict, metrics, hp_file):
    try:
        with open(hp_file, 'w') as f:
            json.dump({'platform': PLATFORM_NAME, 'timestamp': time.strftime("%Y%m%d_%H%M%S"), 'hyperparameters': hp_dict, 'metrics': metrics}, f, indent=2)
        return True
    except Exception:
        return False

def load_tuned_hyperparameters(hp_file):
    if not os.path.exists(hp_file):
        return None
    try:
        with open(hp_file, 'r') as f:
            data = json.load(f)
        return data.get('hyperparameters')
    except Exception:
        return None

def build_hyperparams_dict(experiment_time, beta, learning_rate, epsilon_min, batch_size, discount_factor, target_temp):
    return {'exp': experiment_time, 'beta': beta, 'lr': learning_rate, 'eps_min': epsilon_min, 'batch': batch_size, 'gamma': discount_factor, 'temp': target_temp}

def build_filename_with_hyperparams(module_name, timestamp, hyperparams):
    return f"{module_name}_{timestamp}_{hyperparams['exp']}ep_beta{hyperparams['beta']}_lr{hyperparams['lr']}_eps{hyperparams['eps_min']}_batch{hyperparams['batch']}.csv"

# =============================================================================
# APPLICATION CONFIG
# =============================================================================
class ApplicationConfig:
    def __init__(self):
        self.bots_applications = BOTS_APPLICATIONS
        self.polybench_applications = POLYBENCH_APPLICATIONS
        self.applications_fixed = []
        self.app_list = []
        self._build_applications()
        self.frequency_combinations = FREQUENCY_COMBINATIONS
        self.priority_combinations = PRIORITY_COMBINATIONS
        self.num_cores_list = NUM_CORES_LIST

    def _build_applications(self):
        app_id = 1
        for app_cfg in self.bots_applications:
            self.applications_fixed.append({"id": app_id, "benchmark": app_cfg["benchmark"], "variant": app_cfg["variant"], "input_arg": str(app_cfg["input_arg"]), "app_args": str(app_cfg["input_arg"]), "path": f"bots.sh {app_cfg['benchmark']} {app_cfg['variant']}", "cores": "", "frequencies": [], "priority": None, "action": "profile"})
            self.app_list.append(app_id)
            app_id += 1
        for app_cfg in self.polybench_applications:
            self.applications_fixed.append({"id": app_id, "benchmark": app_cfg["benchmark"], "variant": "", "input_arg": str(app_cfg["input_arg"]), "app_args": str(app_cfg["input_arg"]), "path": f"polybench {app_cfg['benchmark']}", "cores": "", "frequencies": [], "priority": None, "action": "profile"})
            self.app_list.append(app_id)
            app_id += 1
        logging.info(f"Built {len(self.applications_fixed)} applications")

# =============================================================================
# NETWORKING
# =============================================================================
def parse_profiling_data(data):
    try:
        return json.loads(data).get('profiling_data_list', [])
    except json.JSONDecodeError:
        return None

def establish_connection():
    print("Waiting for connection")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((IP_ADDRESS, PORT))
    server_socket.listen(1)
    client_socket, address = server_socket.accept()
    print(f"Connection established with {address[0]}:{address[1]}")
    return client_socket, server_socket

# =============================================================================
# PROFILING
# =============================================================================
def enrich_profiling_data(profiling_data):
    enriched = dict(profiling_data)
    for k, v in {"frequencies": [0], "cores": "1,2,3,4,5", "cache_misses": 0, "branch_misses": 0}.items():
        if k not in enriched:
            enriched[k] = v
    if "total_energy_consumption" not in enriched:
        enriched["total_energy_consumption"] = sum(enriched.get(k, 0.0) for k in ["energy_system_j", "energy_main_j", "energy_cpu_j", "energy_denver_j", "energy_gpu_j", "energy_ddr_j"])
    if "avg_temp_after" not in enriched:
        temps = [enriched.get(f"thermal_zone{i}", 50.0) for i in range(10)]
        enriched["avg_temp_after"] = np.mean([t for t in temps if t > 0]) if any(t > 0 for t in temps) else 50.0
    return enriched

def perform_profiling_phase(client_socket, app_config, timeout=600.0):
    logging.info("Starting Application Profiling Phase")
    profiling_data_list, application_profiles = [], {}

    for app in app_config.applications_fixed:
        application = {'id': app['id'], 'benchmark': app['benchmark'], 'variant': app['variant'], 'input_arg': app['input_arg'], 'app_args': app['input_arg'], 'path': app['path'], 'cores': '', 'frequencies': [], 'priority': None, 'action': 'profile'}
        client_socket.send((json.dumps({'applications': [application]}) + '\n').encode())

        data_received, recv_buffer, start_time = False, '', time.time()
        while not data_received and time.time() - start_time < timeout:
            data = client_socket.recv(4096)
            if data:
                recv_buffer += data.decode()
                while '\n' in recv_buffer:
                    msg, recv_buffer = recv_buffer.split('\n', 1)
                    new_pd_list = parse_profiling_data(msg)
                    if new_pd_list and len(new_pd_list) == 1:
                        data_received = True
                        profiling_data = enrich_profiling_data(new_pd_list[0])
                        app_id = profiling_data.get('application_id', application['id'])
                        mk_onecore, mk_allcore = profiling_data.get('makespan_one_core_frequency_11'), profiling_data.get('makespan_all_cores_frequency_11')
                        profiling_data['parallelism_level'] = mk_onecore / mk_allcore if mk_onecore and mk_allcore and mk_allcore != 0 else 1.0
                        application_profiles[(app_id, str(profiling_data.get('app_args', '')))] = profiling_data
                        profiling_data_list.append(profiling_data)
                        client_socket.send((json.dumps({'status': 'received'}) + '\n').encode())
            else:
                time.sleep(1)

    logging.info(f"Profiling Phase Complete - {len(profiling_data_list)} apps profiled")
    return application_profiles, profiling_data_list

# =============================================================================
# RL MODULE REGISTRY
# =============================================================================
RL_MODULE_REGISTRY = {}

def import_and_register_modules(module_list):
    logging.info("Importing and registering RL modules...")
    for display_name, module_name, function_name in module_list:
        try:
            module = importlib.import_module(module_name)
            RL_MODULE_REGISTRY[display_name] = getattr(module, function_name)
            logging.info(f"Registered: {display_name}")
        except (ImportError, AttributeError) as e:
            logging.warning(f"Could not import {module_name}: {e}")
    logging.info(f"Total modules registered: {len(RL_MODULE_REGISTRY)}")

# =============================================================================
# RUN MODULE EVALUATION
# =============================================================================
def run_module_evaluation(module_name, train_function, client_socket, profiling_data_list, application_profiles, app_config, hyperparams, seed, experiment_time, save_dir, timestamp):
    logging.info(f"\n--- Evaluating {module_name} with seed {seed} ---")
    current_hp = build_hyperparams_dict(experiment_time, hyperparams['beta'], hyperparams['learning_rate'], hyperparams['epsilon_min'], hyperparams['batch_size'], hyperparams['discount_factor'], hyperparams['target_temp'])
    server_name_main = os.path.join(save_dir, build_filename_with_hyperparams(f"{module_name}_seed{seed}", timestamp, current_hp))

    try:
        train_kwargs = {
            'client_socket': client_socket, 'data_keys': DATA_KEYS, 'experiment_time': experiment_time,
            'clock_change_time': hyperparams['clock_change_time'], 'beta': hyperparams['beta'], 'load_model': False,
            'learn_count': hyperparams['model_train_start'], 'plan_count': hyperparams['plan_count'],
            'mem_size': hyperparams['mem_size'], 'learning_rate': hyperparams['learning_rate'],
            'discount_factor': hyperparams['discount_factor'], 'epsilon': hyperparams['epsilon'],
            'epsilon_decay': hyperparams['epsilon_decay'], 'epsilon_min': hyperparams['epsilon_min'],
            'epsilon_start': hyperparams['epsilon_start'], 'epsilon_end': hyperparams['epsilon_end'],
            'reset_learning_rate_value': hyperparams['reset_learning_rate_value'],
            'save_repetition': hyperparams['save_repetition'], 'save_model': False,
            'batch_size': hyperparams['batch_size'], 'agent_train_start': hyperparams['agent_train_start'],
            'target_temp': hyperparams['target_temp'], 'server_name_1': "", 'server_name_2': "",
            'server_name_main': server_name_main, 'profiling_data_list': profiling_data_list,
            'application_profiles': application_profiles, 'applications_fixed': app_config.applications_fixed,
            'priority_combinations': app_config.priority_combinations,
            'frequency_combinations': app_config.frequency_combinations, 'num_cores_list': app_config.num_cores_list,
        }
        if module_name in MODULES_WITH_MODEL_TRAIN_START:
            train_kwargs['model_train_start'] = hyperparams['model_train_start']
            train_kwargs['real_synthetic_ratio'] = hyperparams['real_synthetic_ratio']

        results = train_function(**train_kwargs)
        if results:
            _, makespan_per_app, energy_per_app, temp_per_app = results
            if makespan_per_app and makespan_per_app[0]:
                num_iters = len(makespan_per_app[0])
                return {
                    'module': module_name, 'seed': seed,
                    'makespans': [sum(app[i] for app in makespan_per_app if i < len(app)) for i in range(num_iters)],
                    'energies': [sum(app[i] for app in energy_per_app if i < len(app)) for i in range(num_iters)],
                    'temperatures': [np.mean([app[i] for app in temp_per_app if i < len(app) and app[i] > 0]) if any(app[i] > 0 for app in temp_per_app if i < len(app)) else 50.0 for i in range(num_iters)],
                    'csv_file': server_name_main
                }
        return None
    except Exception as e:
        logging.error(f"{module_name} evaluation failed: {e}", exc_info=True)
        return None

# =============================================================================
# CONVERGENCE PLOT: SA (SAMBRL vs SAMFRL) and MA (MAMBRL vs MAMFRL)
# =============================================================================
def plot_convergence_mb_vs_mf(results, save_path, title="Model-Based vs Model-Free Convergence"):
    """
    Plot convergence comparing MB vs MF for both Single-Agent and Multi-Agent.
    2x3 grid: (SA row, MA row) x (Makespan, Energy, Reward columns)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    sa_modules = ['SAMFRL', 'SAMBRL']
    ma_modules = ['MAMFRL_D3QN', 'MAMBRL_D3QN']
    row_labels = ['Single-Agent (SA)', 'Multi-Agent (MA)']
    module_groups = [sa_modules, ma_modules]

    for row_idx, (modules, row_label) in enumerate(zip(module_groups, row_labels)):
        # Makespan
        ax = axes[row_idx, 0]
        for module_name in modules:
            if module_name not in results or 'aggregate' not in results[module_name]:
                continue
            agg = results[module_name]['aggregate']
            cfg = RL_MODULES.get(module_name, {})
            x = range(len(agg['makespan_mean']))
            ax.plot(x, agg['makespan_mean'], label=cfg.get('display_name', module_name),
                    color=cfg.get('color', 'black'), linestyle=cfg.get('linestyle', '-'), linewidth=2.5)
            ax.fill_between(x, agg['makespan_mean'] - agg['makespan_std'],
                           agg['makespan_mean'] + agg['makespan_std'], alpha=0.2, color=cfg.get('color', 'black'))
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Makespan (s)')
        ax.set_title(f'{row_label}: Makespan')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Energy
        ax = axes[row_idx, 1]
        for module_name in modules:
            if module_name not in results or 'aggregate' not in results[module_name]:
                continue
            agg = results[module_name]['aggregate']
            cfg = RL_MODULES.get(module_name, {})
            x = range(len(agg['energy_mean']))
            ax.plot(x, agg['energy_mean'], label=cfg.get('display_name', module_name),
                    color=cfg.get('color', 'black'), linestyle=cfg.get('linestyle', '-'), linewidth=2.5)
            ax.fill_between(x, agg['energy_mean'] - agg['energy_std'],
                           agg['energy_mean'] + agg['energy_std'], alpha=0.2, color=cfg.get('color', 'black'))
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Energy (J)')
        ax.set_title(f'{row_label}: Energy')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Reward (negative makespan)
        ax = axes[row_idx, 2]
        for module_name in modules:
            if module_name not in results or 'aggregate' not in results[module_name]:
                continue
            agg = results[module_name]['aggregate']
            cfg = RL_MODULES.get(module_name, {})
            reward_mean = -np.array(agg['makespan_mean'])
            x = range(len(reward_mean))
            ax.plot(x, reward_mean, label=cfg.get('display_name', module_name),
                    color=cfg.get('color', 'black'), linestyle=cfg.get('linestyle', '-'), linewidth=2.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title(f'{row_label}: Reward')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved convergence plot to: {save_path}")

# =============================================================================
# COMPUTATIONAL COMPLEXITY TABLE (includes FEDERATED and GraphPerf-RT)
# =============================================================================
def generate_complexity_table(results, n_apps, n_episodes, plan_count, save_path):
    """
    Generate computational complexity comparison table.

    Sample Complexity:
    - FEDERATED: O(n_apps * n_speeds * n_core_configs) - needs profiling table
                 For TX2: n_apps * 12 speeds * 5 core configs = n_apps * 60
    - Model-Free (SAMFRL, MAMFRL): O(N) - N real environment interactions for learning
    - Model-Based (SAMBRL, MAMBRL): O(N_real) real + world model generates synthetic
    - GraphPerf-RT: Zero-shot - only GAT API inference calls, no real samples needed

    Args:
        results: Evaluation results dict
        n_apps: Number of applications
        n_episodes: Number of episodes
        plan_count: Number of synthetic samples per real sample
        save_path: Path to save LaTeX table
    """
    # FEDERATED needs a profiling table: apps × speeds × core_configs
    federated_profiling_samples = n_apps * TX2_NUM_SPEEDS * TX2_NUM_CORE_CONFIGS

    # RL methods need episodes of interaction
    rl_real_samples = n_apps * n_episodes

    # Model-based additionally generates synthetic samples
    mb_synthetic_samples = rl_real_samples * plan_count

    # GraphPerf-RT: zero-shot via GAT, only API calls
    graphperf_samples = 0  # No real environment samples needed

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Computational Complexity Comparison on TX2}",
        "\\label{tab:complexity}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Type & Profiling Samples & Learning Samples & Total \\\\",
        "\\midrule",
    ]

    complexity_data = [
        ('FEDERATED', 'Heuristic', federated_profiling_samples, 0, federated_profiling_samples,
         f'$O(A \\cdot S \\cdot C)$ = {n_apps}$\\times${TX2_NUM_SPEEDS}$\\times${TX2_NUM_CORE_CONFIGS}'),
        ('SAMFRL', 'MF-SA', 0, rl_real_samples, rl_real_samples,
         f'$O(A \\cdot N)$ = {n_apps}$\\times${n_episodes}'),
        ('SAMBRL', 'MB-SA', 0, rl_real_samples, rl_real_samples + mb_synthetic_samples,
         f'$O(A \\cdot N)$ + {plan_count}$\\times$ synthetic'),
        ('MAMFRL\\_D3QN', 'MF-MA', 0, rl_real_samples, rl_real_samples,
         f'$O(A \\cdot N)$ = {n_apps}$\\times${n_episodes}'),
        ('MAMBRL\\_D3QN', 'MB-MA', 0, rl_real_samples, rl_real_samples + mb_synthetic_samples,
         f'$O(A \\cdot N)$ + {plan_count}$\\times$ synthetic'),
        ('GraphPerf-RT', 'Zero-Shot', 0, 0, 0,
         'API calls only'),
    ]

    for name, type_str, prof, learn, total, complexity in complexity_data:
        lines.append(f"{name} & {type_str} & {prof:,} & {learn:,} & {total:,} \\\\")

    lines.extend([
        "\\midrule",
        f"\\multicolumn{{5}}{{l}}{{\\footnotesize A={n_apps} apps, S={TX2_NUM_SPEEDS} speeds, C={TX2_NUM_CORE_CONFIGS} core configs, N={n_episodes} episodes}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    logging.info(f"Saved complexity table to: {save_path}")

    # Console output
    print("\n" + "=" * 90)
    print("COMPUTATIONAL COMPLEXITY COMPARISON")
    print("=" * 90)
    print(f"{'Method':<18} {'Type':<12} {'Profiling':<15} {'Learning':<15} {'Total':<15}")
    print("-" * 90)
    for name, type_str, prof, learn, total, _ in complexity_data:
        name_clean = name.replace('\\_', '_')
        print(f"{name_clean:<18} {type_str:<12} {prof:<15,} {learn:<15,} {total:<15,}")
    print("=" * 90)
    print(f"Parameters: {n_apps} apps, {TX2_NUM_SPEEDS} frequency levels, {TX2_NUM_CORE_CONFIGS} core configs, {n_episodes} episodes")
    print("-" * 90)
    print("Notes:")
    print(f"  - FEDERATED: Requires profiling table of {federated_profiling_samples:,} samples before deployment")
    print(f"  - Model-Free (SAMFRL, MAMFRL): Learn from {rl_real_samples:,} real environment interactions")
    print(f"  - Model-Based (SAMBRL, MAMBRL): Same real samples + {mb_synthetic_samples:,} synthetic from world model")
    print(f"  - GraphPerf-RT: Zero-shot prediction via GAT - latency depends only on API inference time")
    print("=" * 90)

# =============================================================================
# SUMMARY TABLE (performance comparison)
# =============================================================================
def generate_performance_table(results, save_path):
    """Generate performance comparison table (LaTeX format)."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{RL Performance Comparison on Jetson TX2}",
        "\\label{tab:rl_performance}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Type & Final Makespan (s) & Final Energy (J) & Std Dev \\\\",
        "\\midrule",
    ]

    for m in ['SAMFRL', 'SAMBRL', 'MAMFRL_D3QN', 'MAMBRL_D3QN']:
        if m not in results or 'aggregate' not in results[m]:
            continue
        agg = results[m]['aggregate']
        cfg = RL_MODULES.get(m, {})
        mk, en = agg['final_makespan'], agg['final_energy']
        mk_std = np.std(agg['makespan_mean'][-10:]) if len(agg['makespan_mean']) >= 10 else 0
        type_str = f"{'MA' if cfg.get('agent_type') == 'multi' else 'SA'}-{'MB' if 'model-based' in cfg.get('type', '') else 'MF'}"
        lines.append(f"{cfg.get('display_name', m)} & {type_str} & {mk:.3f} & {en:.3f} & {mk_std:.3f} \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    logging.info(f"Saved performance table to: {save_path}")

def generate_csv_results(results, save_path):
    """Save results to CSV."""
    rows = []
    for m, data in results.items():
        if 'aggregate' not in data:
            continue
        agg = data['aggregate']
        cfg = RL_MODULES.get(m, {})
        rows.append({
            'module': m, 'display_name': cfg.get('display_name', m),
            'type': cfg.get('type', 'unknown'), 'agent_type': cfg.get('agent_type', 'unknown'),
            'final_makespan': agg['final_makespan'], 'final_energy': agg['final_energy'],
            'makespan_std': np.std(agg['makespan_mean'][-10:]) if len(agg['makespan_mean']) >= 10 else 0,
        })
    pd.DataFrame(rows).to_csv(save_path, index=False)
    logging.info(f"Saved CSV results to: {save_path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='LIVE RL Baseline Evaluation for TX2')
    parser.add_argument('--task', type=str, default='all', choices=['all', 'convergence', 'complexity'])
    parser.add_argument('--n_episodes', type=int, default=MAIN_EXPERIMENT_TIME)
    parser.add_argument('--n_seeds', type=int, default=N_SEEDS)
    parser.add_argument('--beta', type=float, default=DEFAULT_HYPERPARAMETERS['beta'])
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--modules', type=str, nargs='+', default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print("LIVE RL BASELINE EVALUATION FOR JETSON TX2")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Episodes: {args.n_episodes}, Seeds: {args.n_seeds}")
    print(f"HP Tuning: {args.tune}")
    print("=" * 80)
    print("CONVERGENCE ANALYSIS: MB vs MF for SA and MA")
    print("  - Single-Agent: SAMBRL (MB) vs SAMFRL (MF)")
    print("  - Multi-Agent: MAMBRL_D3QN (MB) vs MAMFRL_D3QN (MF)")
    print("FEDERATED included only in computational complexity table")
    print("=" * 80)

    # Import modules based on task
    if args.task == 'complexity':
        import_and_register_modules(ALL_MODULES)
    else:
        import_and_register_modules(CONVERGENCE_MODULES)

    if not RL_MODULE_REGISTRY:
        logging.error("No RL modules registered!")
        return

    modules_to_evaluate = args.modules if args.modules else list(RL_MODULE_REGISTRY.keys())
    modules_to_evaluate = [m for m in modules_to_evaluate if m in RL_MODULE_REGISTRY]
    logging.info(f"Modules to evaluate: {modules_to_evaluate}")

    hyperparams = dict(DEFAULT_HYPERPARAMETERS)
    hyperparams['beta'] = args.beta
    loaded_hp = load_tuned_hyperparameters(TUNED_HP_FILE)
    if loaded_hp and not args.tune:
        hyperparams.update(loaded_hp)

    app_config = ApplicationConfig()
    n_apps = len(app_config.applications_fixed)

    # If only complexity table requested, generate without running experiments
    if args.task == 'complexity':
        generate_complexity_table({}, n_apps, args.n_episodes, hyperparams['plan_count'],
                                  os.path.join(TABLES_DIR, f'complexity_table_TX2_{timestamp}.tex'))
        return

    # Establish connection for live evaluation
    client_socket, server_socket = establish_connection()

    try:
        application_profiles, profiling_data_list = perform_profiling_phase(client_socket, app_config)
        if not profiling_data_list:
            logging.error("No profiling data collected!")
            return

        # HP TUNING
        if args.tune and HP_TUNING_MODULE in RL_MODULE_REGISTRY:
            logging.info(f"HP TUNING WITH {HP_TUNING_MODULE}")
            tuning_results = []
            tuning_fn = RL_MODULE_REGISTRY[HP_TUNING_MODULE]
            HP_SEARCH_SPACE = {k: HP_SEARCH_VALUES[k] if HP_TUNE_CONFIG.get(k) else [HP_SEARCH_VALUES[k][len(HP_SEARCH_VALUES[k])//2]] for k in HP_SEARCH_VALUES}
            for hp_combo in product(*[HP_SEARCH_SPACE[k] for k in HP_SEARCH_SPACE]):
                hp_dict = dict(zip(HP_SEARCH_SPACE.keys(), hp_combo))
                test_hp = {**hyperparams, **hp_dict}
                result = run_module_evaluation(HP_TUNING_MODULE, tuning_fn, client_socket, profiling_data_list, application_profiles, app_config, test_hp, 42, HP_TUNING_EXPERIMENT_TIME, SAVE_DIR, timestamp)
                if result:
                    start_idx = len(result['makespans']) // 2
                    tuning_results.append({'hyperparams': hp_dict, 'metrics': {'avg_makespan': np.mean(result['makespans'][start_idx:])}})
                time.sleep(3)
            if tuning_results:
                best = min(tuning_results, key=lambda x: x['metrics']['avg_makespan'])
                hyperparams.update(best['hyperparams'])
                save_tuned_hyperparameters(best['hyperparams'], best['metrics'], TUNED_HP_FILE)

        # MAIN EVALUATION
        logging.info("MAIN EVALUATION PHASE")
        all_results = {m: {'seeds': {}} for m in modules_to_evaluate}

        for module_name in modules_to_evaluate:
            train_fn = RL_MODULE_REGISTRY[module_name]
            for seed_idx, seed in enumerate(RANDOM_SEEDS[:args.n_seeds]):
                logging.info(f"--- {module_name} Seed {seed_idx + 1}/{args.n_seeds}: {seed} ---")
                result = run_module_evaluation(module_name, train_fn, client_socket, profiling_data_list, application_profiles, app_config, hyperparams, seed, args.n_episodes, SAVE_DIR, timestamp)
                if result:
                    all_results[module_name]['seeds'][seed] = result
                cleanup_memory()
                time.sleep(5)

        # AGGREGATE RESULTS
        for m in modules_to_evaluate:
            if not all_results[m]['seeds']:
                continue
            all_mk = [s['makespans'] for s in all_results[m]['seeds'].values()]
            all_en = [s['energies'] for s in all_results[m]['seeds'].values()]
            max_len = max(len(x) for x in all_mk)
            all_mk = np.array([x + [x[-1]]*(max_len - len(x)) for x in all_mk])
            all_en = np.array([x + [x[-1]]*(max_len - len(x)) for x in all_en])
            all_results[m]['aggregate'] = {
                'makespan_mean': np.mean(all_mk, axis=0), 'makespan_std': np.std(all_mk, axis=0),
                'energy_mean': np.mean(all_en, axis=0), 'energy_std': np.std(all_en, axis=0),
                'final_makespan': np.mean(all_mk[:, -10:]), 'final_energy': np.mean(all_en[:, -10:])
            }

        # GENERATE OUTPUTS
        if args.task in ['all', 'convergence']:
            plot_convergence_mb_vs_mf(all_results, os.path.join(FIGURES_DIR, f'convergence_MB_vs_MF_TX2_{timestamp}.png'),
                                      title="Model-Based vs Model-Free Convergence (TX2)")

        # Always generate complexity table
        generate_complexity_table(all_results, n_apps, args.n_episodes, hyperparams['plan_count'],
                                  os.path.join(TABLES_DIR, f'complexity_table_TX2_{timestamp}.tex'))

        generate_performance_table(all_results, os.path.join(TABLES_DIR, f'performance_table_TX2_{timestamp}.tex'))
        generate_csv_results(all_results, os.path.join(TABLES_DIR, f'results_TX2_{timestamp}.csv'))

        # SAVE RAW RESULTS
        with open(os.path.join(RESULTS_DIR, f'results_TX2_{timestamp}.json'), 'w') as f:
            def convert(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list): return [convert(v) for v in obj]
                elif isinstance(obj, (np.int64, np.int32)): return int(obj)
                elif isinstance(obj, (np.float64, np.float32)): return float(obj)
                return obj
            json.dump(convert(all_results), f, indent=2)

        # PRINT SUMMARY
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY: Model-Based vs Model-Free")
        print("=" * 80)
        print(f"{'Module':<20} {'Type':<10} {'Final Makespan':<18} {'Final Energy':<18}")
        print("-" * 70)
        for m in modules_to_evaluate:
            if m in all_results and 'aggregate' in all_results[m]:
                agg = all_results[m]['aggregate']
                cfg = RL_MODULES.get(m, {})
                type_str = f"{'MA' if cfg.get('agent_type') == 'multi' else 'SA'}-{'MB' if 'model-based' in cfg.get('type', '') else 'MF'}"
                print(f"{m:<20} {type_str:<10} {agg['final_makespan']:<18.3f} {agg['final_energy']:<18.3f}")

        # Show improvement of MB over MF
        print("\n" + "-" * 70)
        print("Model-Based Improvement over Model-Free:")
        for sa_mf, sa_mb in [('SAMFRL', 'SAMBRL'), ('MAMFRL_D3QN', 'MAMBRL_D3QN')]:
            if sa_mf in all_results and 'aggregate' in all_results[sa_mf] and sa_mb in all_results and 'aggregate' in all_results[sa_mb]:
                mf_mk = all_results[sa_mf]['aggregate']['final_makespan']
                mb_mk = all_results[sa_mb]['aggregate']['final_makespan']
                imp = (mf_mk - mb_mk) / mf_mk * 100
                agent_type = 'SA' if sa_mf == 'SAMFRL' else 'MA'
                print(f"  {agent_type}: {sa_mb} vs {sa_mf}: {imp:+.1f}% makespan improvement")

        print("=" * 80)
        print(f"Results saved to: {RESULTS_DIR}")

    finally:
        try:
            client_socket.close()
            server_socket.close()
        except:
            pass

    return all_results

if __name__ == '__main__':
    main()
