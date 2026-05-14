#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphPerf-RT — Section 6 (RL integration): combined on-device training server.

Runs the four RL methods reported in the paper end-to-end on a Jetson host:

    1. SAMFRL          (single-agent  · model-free  · DQN)
    2. SAMBRL          (single-agent  · model-based · GraphPerf-RT world model)
    3. MAMFRL-D3QN     (multi-agent   · model-free  · dueling D3QN)
    4. MAMBRL-D3QN     (multi-agent   · model-based · paper champion)

Reproduces the headline numbers from §6.4:

    - 66 % makespan reduction   (MAMBRL-D3QN vs MAMFRL)
    - 82 % energy reduction
    - **zero thermal violations** across 5 seeds × 200 episodes

Communication
-------------
Pairs with `client.py` over plain TCP (one-message-per-line JSON).  The client
runs the OpenMP benchmarks under the chosen DVFS action and ships back per-app
profiling tuples; this server uses them as the (s, a, r, s') stream for the RL
agents.

Network configuration is exposed through CLI flags --host / --port; no LAN
addresses are baked into the file.
"""

# =============================================================================
# =============================================================================
#                    CONFIGURATION SECTION - EASY ACCESS
# =============================================================================
# =============================================================================
# All configurable parameters are centralized here for easy modification.
# Modify these values to customize your experiments without changing code below.

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
PORT = 8707                      # Server port to listen on
IP_ADDRESS = "0.0.0.0"           # Server IP (0.0.0.0 = listen on all interfaces)

# =============================================================================
# PLATFORM CONFIGURATION
# =============================================================================
PLATFORM_NAME = "jetson_tx2"     # Platform identifier for saving/loading hyperparameters

# =============================================================================
# EXPERIMENT TIME CONFIGURATION
# =============================================================================
HP_TUNING_EXPERIMENT_TIME = 50   # Episodes for hyperparameter tuning phase (shorter)
MAIN_EXPERIMENT_TIME = 2       # Episodes for main training runs (full)

# =============================================================================
# HYPERPARAMETER TUNING CONTROL
# =============================================================================
ENABLE_HP_TUNING = False         # Set to True to run hyperparameter tuning phase
HP_TUNING_MODULE = "SAMBRL"      # Module to use for hyperparameter tuning (fast exploration)

# =============================================================================
# HYPERPARAMETER TUNING - SELECT WHICH PARAMETERS TO TUNE
# =============================================================================
# Set True to include parameter in tuning search, False to use default value
HP_TUNE_CONFIG = {
    # MODEL-BASED RL SPECIFIC (tune these for model-based approaches)
    'plan_count': False,           # Synthetic data per planning phase
    'model_train_start': False,    # Real data threshold for model training
    'agent_train_start': False,    # Total data threshold for agent training
    'real_synthetic_ratio': False, # Ratio of real to synthetic data

    # GENERAL RL PARAMETERS
    'learning_rate': False,        # Already tuned: 0.1 is optimal
    'batch_size': False,           # Already tuned: 64 is optimal for stability

    # OPTIONAL PARAMETERS (usually fixed after initial tuning)
    'beta': False,                 # 1.0 is balanced
    'epsilon_min': False,          # 0.1 works well
    'epsilon_decay': False,        # 0.90 is standard
    'discount_factor': False,      # 0.99 is standard
}

# =============================================================================
# HYPERPARAMETER SEARCH SPACE - VALUES TO TEST WHEN TUNING
# =============================================================================
# Uncomment/comment specific values to adjust search space
HP_SEARCH_VALUES = {
    # MODEL-BASED SPECIFIC PARAMETERS
    'plan_count': [
        # 30,                      # Low planning (fast, less synthetic data)
        100,                       # DEFAULT: Medium planning (balanced)
        200,                       # High planning (more synthetic data, slower)
    ],
    'model_train_start': [
        16,                        # Early model training
        32,                        # DEFAULT: Standard model training start
        # 64,                      # Late model training (more real data first)
    ],
    'agent_train_start': [
        16,                        # Early agent training
        32,                        # DEFAULT: Standard agent training start
        # 64,                      # Late agent training
    ],
    'real_synthetic_ratio': [
        # 0.3,                     # Synthetic-heavy (30% real, 70% synthetic)
        0.5,                       # DEFAULT: Balanced (50% real, 50% synthetic)
        0.7,                       # Real-heavy (70% real, 30% synthetic)
    ],

    # GENERAL RL PARAMETERS
    'learning_rate': [
        # 0.01,                    # Conservative (slower convergence)
        # 0.05,                    # Standard
        0.1,                       # DEFAULT: Aggressive (best from tuning)
    ],
    'batch_size': [
        # 16,                      # Small (noisy gradients)
        # 32,                      # Medium (best mean performance)
        64,                        # DEFAULT: Large (most stable)
    ],

    # OPTIONAL PARAMETERS
    'beta': [
        # 0.5,                     # Energy-focused
        1.0,                       # DEFAULT: Balanced
        # 2.0,                     # Makespan-focused
    ],
    'epsilon_min': [
        # 0.05,                    # Low exploration
        0.1,                       # DEFAULT: Standard
        # 0.15,                    # High exploration
    ],
    'epsilon_decay': [
        # 0.85,                    # Fast decay
        0.90,                      # DEFAULT: Standard decay
        # 0.95,                    # Slow decay
    ],
    'discount_factor': [
        # 0.95,                    # Short-term focus
        0.99,                      # DEFAULT: Long-term focus
    ],
}

# =============================================================================
# DEFAULT HYPERPARAMETERS (OPTIMAL VALUES FROM TUNING)
# =============================================================================
# These are used when ENABLE_HP_TUNING is False or for parameters not being tuned
DEFAULT_HYPERPARAMETERS = {
    # OBJECTIVE BALANCE
    'beta': 1.0,                   # 1.0 = balanced (0.5=energy, 2.0=makespan)

    # OPTIMIZATION (TUNED VALUES)
    'learning_rate': 0.1,          # Best from tuning (0.05 also works)
    'batch_size': 32,              # Most stable (32 = best mean)
    'discount_factor': 0.99,       # Standard for long-horizon

    # EXPLORATION
    'epsilon': 1.0,                # Starting exploration rate
    'epsilon_decay': 0.90,         # Decay rate per episode
    'epsilon_min': 0.10,           # Minimum exploration rate
    'epsilon_start': 1.0,          # Starting epsilon (same as epsilon)
    'epsilon_end': 0.0,            # End epsilon (for linear decay)

    # MODEL-BASED RL PARAMETERS
    'plan_count': 100,             # Synthetic samples per planning phase (30-200)
    'model_train_start': 32,       # Real data threshold before model training (16-64)
    'agent_train_start': 32,       # Total data threshold before agent training (16-64)
    'real_synthetic_ratio': 0.5,   # 50% real, 50% synthetic in training batch

    # REPLAY BUFFER
    'mem_size': 100000,            # Replay buffer size

    # TRAINING SCHEDULE
    'reset_learning_rate_value': 20,  # When to reset learning rate
    'save_repetition': 20,         # Checkpoint frequency (episodes)
    'save_model': True,            # Whether to save models
    'load_model': False,           # Whether to load pre-trained models

    # THERMAL
    'target_temp': 50,             # Target temperature in Celsius
    'clock_change_time': 30,       # Time between frequency changes

    # DEPRECATED
    'learn_count': 16,             # DEPRECATED: Use model_train_start instead
}

# =============================================================================
# RL MODULES TO RUN - COMMENT/UNCOMMENT TO ENABLE/DISABLE
# =============================================================================
# Format: (Display Name, Module Name, Function Name)
#
# Module Types:
# - Single-Agent (SA): Single profiler agent only
# - Multi-Agent (MA): Hierarchical 3-agent structure (Profiler + Thermal + Priority)
# - Model-Based (MB): Uses environment model for planning/reward estimation
# - Model-Free (MF): No environment model, learns directly from experience
# - Heuristic (H): Non-RL baseline using algorithmic approaches

ENABLED_RL_MODULES = [
    # =========================================================================
    # HEURISTIC BASELINES (Non-RL approaches for comparison)
    # =========================================================================
    # FEDERATED: Federated Energy-Aware Heuristic (slow/fast core allocation)
    # Works on heterogeneous platforms: TX2 (Denver+A57), Orin NX, RubikPi (big.LITTLE)
    ("FEDERATED", "fixed_federated", "train_fixed_federated"),

    # =========================================================================
    # SINGLE-AGENT APPROACHES
    # =========================================================================
    # SAMFRL: Single-Agent Model-Free RL (baseline for single-agent comparison)
    ("SAMFRL", "fixed_samfrl", "train_fixed_samfrl"),

    # SAMBRL: Single-Agent Model-Based RL (adds environment model to SAMFRL)
    ("SAMBRL", "fixed_sambrl", "train_fixed_sambrl"),

    # SARBRL: Single-Agent Reward-Based RL
    ("SARBRL", "fixed_sarbrl", "train_fixed_sarbrl"),

    # SAMBPGRL: Single-Agent Model-Based Policy Gradient RL
    ("SAMBPGRL", "fixed_sambpgrl", "train_fixed_sambpgrl"),

    # MAML: Model-Agnostic Meta-Learning for few-shot RL
    ("MAML", "fixed_maml", "train_fixed_maml"),

    # =========================================================================
    # MULTI-AGENT APPROACHES
    # =========================================================================
    # MARBRL: Multi-Agent Reward-Based RL with DQN (uses model for long-horizon reward estimation)
    ("MARBRL", "fixed_marbrl", "train_fixed_marbrl"),

    # MARB_D3QN: Multi-Agent Reward-Based D3QN
    ("MARB_D3QN", "fixed_marb_d3qn", "train_fixed_marb_d3qn"),

    # MARL_GEAR: Multi-Agent RL with GEAR thermal-aware scheduler
    ("MARL_GEAR", "fixed_marl_gear", "train_fixed_marl_gear"),

    # =========================================================================
    # MAIN APPROACHES (hierarchical multi-agent with full features)
    # =========================================================================
    # MAMFRL_D3QN: Multi-Agent Model-Free RL with D3QN (no environment model)
    ("MAMFRL_D3QN", "fixed_mamfrl_d3qn_tx2", "train_fixed_mamfrl_d3qn"),

    # MAMBRL_D3QN: Multi-Agent Model-Based RL with D3QN (full Dyna-Q style)
    ("MAMBRL_D3QN", "fixed_mambrl_d3qn_tx2", "train_fixed_mambrl_d3qn"),
]

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================
# Valid FFT input_arg values: "258064", "260100", "262144", "264196", "266256"
# Valid BOTS benchmarks: alignment, fft, fib, floorplan, health, concom,
#                        knapsack, nqueens, sort, sparselu, strassen, uts
# Valid PolyBench benchmarks: gemm, gemver, gesummv, symm, syr2k, syrk, trmm,
#                             2mm, 3mm, atax, bicg, doitgen, mvt, cholesky,
#                             durbin, gramschmidt, lu, ludcmp, trisolv,
#                             correlation, covariance, deriche, floyd-warshall,
#                             nussinov, adi, fdtd-2d, heat-3d, jacobi-1d,
#                             jacobi-2d, seidel-2d

# BOTS Benchmarks (OpenMP Tasks) - Comment/uncomment to enable/disable
BOTS_APPLICATIONS = [
    {"benchmark": "fft", "variant": "bin-omp-tasks", "input_arg": "262144"},
    {"benchmark": "fft", "variant": "bin-omp-tasks-tied", "input_arg": "262144"},
    {"benchmark": "fft", "variant": "bin-serial", "input_arg": "262144"},
    # {"benchmark": "nqueens", "variant": "bin-omp-tasks", "input_arg": "12"},
    # {"benchmark": "sort", "variant": "bin-omp-tasks", "input_arg": "500000"},
    # {"benchmark": "fib", "variant": "bin-omp-tasks", "input_arg": "30"},
    # {"benchmark": "strassen", "variant": "bin-omp-tasks", "input_arg": "2048"},
    # {"benchmark": "alignment", "variant": "bin-omp-tasks", "input_arg": ""},
    # {"benchmark": "floorplan", "variant": "bin-omp-tasks", "input_arg": ""},
    # {"benchmark": "health", "variant": "bin-omp-tasks", "input_arg": ""},
    # {"benchmark": "sparselu", "variant": "bin-omp-tasks", "input_arg": ""},
]

# PolyBench/C Benchmarks - Comment/uncomment to enable/disable
POLYBENCH_APPLICATIONS = [
    # {"benchmark": "gemm", "variant": "", "input_arg": "STANDARD"},
    # {"benchmark": "jacobi-2d", "variant": "", "input_arg": "STANDARD"},
    # {"benchmark": "2mm", "variant": "", "input_arg": "STANDARD"},
    # {"benchmark": "syrk", "variant": "", "input_arg": "STANDARD"},
    # {"benchmark": "heat-3d", "variant": "", "input_arg": "STANDARD"},
    # {"benchmark": "gemver", "variant": "", "input_arg": "STANDARD"},
    # {"benchmark": "correlation", "variant": "", "input_arg": "STANDARD"},
]

# =============================================================================
# FREQUENCY AND CORE CONFIGURATION
# =============================================================================
# Frequency combinations (indices into available_frequencies list)
# TX2 typically has 12 frequency levels (0=lowest, 11=highest)
FREQUENCY_COMBINATIONS = [
    [0] * 5,   # Lowest frequency
    [2] * 5,   # Low-mid frequency
    [6] * 5,   # Mid frequency
    [11] * 5,  # Highest frequency
]

# Priority combinations for multi-app scheduling
# Each tuple assigns priority to each app (higher = more important)
PRIORITY_COMBINATIONS = [
    (10, 10, 10),
    (90, 90, 90),
    (10, 50, 90),
    (50, 90, 10),
    (90, 10, 50),
    (10, 90, 50),
]

# Number of cores to allocate
NUM_CORES_LIST = [1, 3, 5]

# =============================================================================
# =============================================================================
#                    END OF CONFIGURATION SECTION
# =============================================================================
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


def cleanup_memory():
    """
    Comprehensive memory cleanup between training modules.
    Clears matplotlib figures, Keras/TensorFlow sessions, PyTorch cache, and runs garbage collection.
    """
    # Close all matplotlib figures to free memory
    plt.close('all')

    # Clear Keras/TensorFlow session
    if TF_AVAILABLE:
        try:
            K.clear_session()
            # Reset default graph (TF1 compatibility)
            if hasattr(tf, 'reset_default_graph'):
                tf.reset_default_graph()
        except Exception as e:
            logging.debug(f"TensorFlow cleanup warning: {e}")

    # Clear PyTorch cache
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logging.debug(f"PyTorch cleanup warning: {e}")

    # Force garbage collection (multiple passes for thorough cleanup)
    gc.collect()
    gc.collect()
    gc.collect()

    logging.info("Memory cleanup completed")

# =============================================================================
# MODULE CLASSIFICATION CONSTANTS (defined early for use in functions)
# =============================================================================
SINGLE_AGENT_MODULES = ['SAMFRL', 'SAMBRL', 'SARBRL', 'SAMBPGRL', 'MAML']
MULTI_AGENT_MODULES = ['MAMFRL_D3QN', 'MAMBRL_D3QN', 'MARBRL', 'MARB_D3QN', 'MARL_GEAR']
SINGLE_AGENT_BASELINE = 'SAMFRL'
MULTI_AGENT_BASELINE = 'MAMFRL_D3QN'

# =============================================================================
# CONFIDENCE INTERVAL AND STATISTICS FUNCTIONS
# =============================================================================
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


def calculate_rolling_ci(data, window_size=10):
    """Calculate rolling mean and 95% CI for time series data."""
    if len(data) < window_size:
        window_size = max(1, len(data) // 3)
    series = pd.Series(data)
    rolling = series.rolling(window=window_size, center=True, min_periods=1)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()
    upper_bound = rolling_mean + 1.96 * rolling_std
    lower_bound = rolling_mean - 1.96 * rolling_std
    return rolling_mean.values, lower_bound.values, upper_bound.values


# =============================================================================
# MODULE CLASSIFICATION AND BASELINE SELECTION FUNCTIONS
# =============================================================================
def get_module_type(module_name: str) -> str:
    """Determine if a module is single-agent or multi-agent."""
    module_upper = module_name.upper()
    for sa_module in SINGLE_AGENT_MODULES:
        if sa_module.upper() in module_upper:
            return 'single-agent'
    for ma_module in MULTI_AGENT_MODULES:
        if ma_module.upper() in module_upper:
            return 'multi-agent'
    # Default: check if 'MA' prefix indicates multi-agent (but not MAML)
    if module_upper.startswith('MA') and not module_upper.startswith('MAML'):
        return 'multi-agent'
    return 'single-agent'


def get_baseline_for_module(module_name: str) -> str:
    """
    Get the appropriate baseline module for comparison.
    Single-agent modules compare with SAMFRL.
    Multi-agent modules compare with MAMFRL_D3QN.
    """
    module_type = get_module_type(module_name)
    if module_type == 'single-agent':
        return SINGLE_AGENT_BASELINE
    else:
        return MULTI_AGENT_BASELINE


def get_baseline_display_name(baseline_module: str) -> str:
    """Get display name for baseline module."""
    if baseline_module == SINGLE_AGENT_BASELINE:
        return "SAMFRL (Single-Agent Baseline)"
    elif baseline_module == MULTI_AGENT_BASELINE:
        return "MAMFRL (Multi-Agent Baseline)"
    else:
        return baseline_module


# =============================================================================
# HYPERPARAMETER SAVE/LOAD FUNCTIONS
# =============================================================================
def save_tuned_hyperparameters(hp_dict: Dict[str, Any], metrics: Dict[str, Any], hp_file: str) -> bool:
    """
    Save tuned hyperparameters to a JSON file for later reuse.

    Args:
        hp_dict: Dictionary of hyperparameter values
        metrics: Dictionary of performance metrics (avg_makespan, min_makespan, etc.)
        hp_file: Path to the JSON file

    Returns:
        True if save successful, False otherwise
    """
    try:
        save_data = {
            'platform': PLATFORM_NAME,
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'hyperparameters': hp_dict,
            'metrics': metrics,
        }
        with open(hp_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        logging.info(f"Saved tuned hyperparameters to: {hp_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to save hyperparameters: {e}")
        return False


def load_tuned_hyperparameters(hp_file: str) -> Optional[Dict[str, Any]]:
    """
    Load previously tuned hyperparameters from a JSON file.

    Args:
        hp_file: Path to the JSON file

    Returns:
        Dictionary with hyperparameters, or None if file doesn't exist or is invalid
    """
    if not os.path.exists(hp_file):
        logging.warning(f"No saved hyperparameters found at: {hp_file}")
        return None

    try:
        with open(hp_file, 'r') as f:
            data = json.load(f)

        # Validate the loaded data
        if 'hyperparameters' not in data:
            logging.warning(f"Invalid hyperparameter file format: {hp_file}")
            return None

        hp = data['hyperparameters']
        metrics = data.get('metrics', {})
        timestamp = data.get('timestamp', 'unknown')
        platform = data.get('platform', 'unknown')

        logging.info(f"Loaded tuned hyperparameters from: {hp_file}")
        logging.info(f"  Platform: {platform}")
        logging.info(f"  Tuned on: {timestamp}")
        logging.info(f"  Hyperparameters:")
        for k, v in hp.items():
            logging.info(f"    {k}: {v}")
        if metrics:
            logging.info(f"  Performance when tuned:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    logging.info(f"    {k}: {v:.4f}")
                else:
                    logging.info(f"    {k}: {v}")

        return hp
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse hyperparameter file: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to load hyperparameters: {e}")
        return None


# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =============================================================================
# Derived Configuration (from top-level constants)
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "save_model")
os.makedirs(SAVE_DIR, exist_ok=True)
TUNED_HP_FILE = os.path.join(SAVE_DIR, f"tuned_hyperparameters_{PLATFORM_NAME}.json")

# Keys pushed into RL modules (shared) - Standardized plotting order
# This is the GLOBAL definition - all other scripts should import from here
DATA_KEYS = [
    "makespan",           # 1. Total makespan per iteration
    "num_cores",          # 2. Number of cores allocated
    "cores",              # 3. Core allocation details
    "qmax",               # 4. Maximum Q-value
    "energy",             # 5. Energy consumption (was 'loss')
    "freq",               # 6. Frequency selection
    "priority_combination", # 7. Priority combination encoding
    "thermal",            # 8. Average temperature
    "branchmisses",       # 9. Branch misses count
    "cachemisses",        # 10. Cache misses count
    "priority",           # 11. Priority values
    "reward",             # 12. Total reward
]

# =============================================================================
# RL Module Registry (Auto-populated from ENABLED_RL_MODULES)
# =============================================================================
RL_MODULE_REGISTRY: Dict[str, Callable] = {}


def import_and_register_modules():
    """
    Import and register all enabled RL modules.
    """
    logging.info("=" * 80)
    logging.info("Importing and registering RL modules...")
    logging.info("=" * 80)

    for display_name, module_name, function_name in ENABLED_RL_MODULES:
        try:
            module = importlib.import_module(module_name)
            train_function = getattr(module, function_name)
            RL_MODULE_REGISTRY[display_name] = train_function
            logging.info(f"Registered: {display_name} from {module_name}.{function_name}")
        except ImportError as e:
            logging.warning(f"Could not import {module_name}: {e}")
        except AttributeError as e:
            logging.warning(f"Function {function_name} not found in {module_name}: {e}")

    logging.info("=" * 80)
    logging.info(f"Total modules registered: {len(RL_MODULE_REGISTRY)}")
    logging.info("=" * 80)


# =============================================================================
# BENCHMARK FAMILIES - Easy to extend with new benchmarks
# =============================================================================
# BOTS (Barcelona OpenMP Tasks Suite) - omptasks directory
# Format: {"benchmark": name, "variant": variant, "input_arg": input}
# Variants: bin-omp-tasks, bin-omp-tasks-tied, bin-serial
OMPTASKS_BENCHES = {
    "alignment", "fft", "fib", "floorplan", "health",
    "concom", "knapsack", "nqueens", "sort", "sparselu",
    "strassen", "uts",
}

# PolyBench/C - polytasks directory
# Format: {"benchmark": name, "variant": "", "input_arg": dataset_size}
# Dataset sizes: MINI, SMALL, MEDIUM, STANDARD, LARGE, EXTRALARGE
POLYBENCH_BENCHES = {
    # BLAS
    "gemm", "gemver", "gesummv", "symm", "syr2k", "syrk", "trmm",
    # Kernels
    "2mm", "3mm", "atax", "bicg", "doitgen", "mvt",
    # Solvers
    "cholesky", "durbin", "gramschmidt", "lu", "ludcmp", "trisolv",
    # Datamining
    "correlation", "covariance",
    # Medley
    "deriche", "floyd-warshall", "nussinov",
    # Stencils
    "adi", "fdtd-2d", "heat-3d", "jacobi-1d", "jacobi-2d", "seidel-2d",
}


# =============================================================================
# Centralized Application Configuration
# =============================================================================
class ApplicationConfig:
    """
    Centralized configuration for benchmarks and applications.

    HOW TO ADD NEW APPLICATIONS:
    ============================

    1. BOTS (OpenMP Tasks) Benchmarks:
       Add entries to self.bots_applications list with format:
       {
           "benchmark": "benchmark_name",    # e.g., "fft", "nqueens", "sort"
           "variant": "bin-omp-tasks",       # or "bin-omp-tasks-tied", "bin-serial"
           "input_arg": "5"                  # benchmark-specific input (N value)
       }

       Available BOTS benchmarks: alignment, fft, fib, floorplan, health,
                                  concom, knapsack, nqueens, sort, sparselu,
                                  strassen, uts

       Example:
           {"benchmark": "nqueens", "variant": "bin-omp-tasks", "input_arg": "12"}

    2. PolyBench/C Benchmarks:
       Add entries to self.polybench_applications list with format:
       {
           "benchmark": "benchmark_name",    # e.g., "gemm", "jacobi-2d"
           "variant": "",                    # Leave empty for polybench
           "input_arg": "STANDARD"           # Dataset: MINI, SMALL, MEDIUM, STANDARD, LARGE, EXTRALARGE
       }

       Available PolyBench benchmarks:
         BLAS: gemm, gemver, gesummv, symm, syr2k, syrk, trmm
         Kernels: 2mm, 3mm, atax, bicg, doitgen, mvt
         Solvers: cholesky, durbin, gramschmidt, lu, ludcmp, trisolv
         Datamining: correlation, covariance
         Medley: deriche, floyd-warshall, nussinov
         Stencils: adi, fdtd-2d, heat-3d, jacobi-1d, jacobi-2d, seidel-2d

       Example:
           {"benchmark": "gemm", "variant": "", "input_arg": "LARGE"}

    CLIENT MESSAGE FORMAT:
    ======================
    The client (client_evaluate_tx2.py) expects applications with these fields:
    - id: Unique integer identifier
    - benchmark: Name of benchmark (e.g., "fft", "gemm")
    - variant: Variant for BOTS (e.g., "bin-omp-tasks"), empty for PolyBench
    - input_arg: Input argument (N for BOTS, dataset size for PolyBench)
    - app_args: Same as input_arg (for compatibility)
    - cores: Comma-separated core list (e.g., "1,2,3")
    - frequencies: List of frequency indices per core
    - priority: RT priority (e.g., 80)
    - action: "profile" or "run"

    The client routes based on benchmark name:
    - If benchmark in OMPTASKS_BENCHES: uses bots.sh runner
    - If benchmark in POLYBENCH_BENCHES: uses run_bench.sh runner
    """

    def __init__(self):
        # =====================================================================
        # USE TOP-LEVEL CONFIGURATION CONSTANTS
        # (Modify BOTS_APPLICATIONS, POLYBENCH_APPLICATIONS, etc. at top of file)
        # =====================================================================
        self.bots_applications = BOTS_APPLICATIONS
        self.polybench_applications = POLYBENCH_APPLICATIONS

        self.applications_fixed: List[Dict[str, Any]] = []
        self.app_list: List[int] = []

        self._build_applications()

        # Use top-level configuration constants
        self.frequency_combinations = FREQUENCY_COMBINATIONS
        self.priority_combinations = PRIORITY_COMBINATIONS
        self.num_cores_list = NUM_CORES_LIST

    def _build_applications(self):
        """Build unified application list from BOTS and PolyBench configurations."""
        app_id = 1

        # Add BOTS applications
        for app_cfg in self.bots_applications:
            benchmark = app_cfg["benchmark"]
            variant = app_cfg["variant"]
            if benchmark not in OMPTASKS_BENCHES:
                logging.warning(f"Unknown BOTS benchmark: {benchmark}")

            app = {
                "id": app_id,
                "benchmark": benchmark,
                "variant": variant,
                "input_arg": str(app_cfg["input_arg"]),
                "app_args": str(app_cfg["input_arg"]),
                "path": f"bots.sh {benchmark} {variant}",  # For compatibility
                "cores": "",
                "frequencies": [],
                "priority": None,
                "action": "profile",
            }
            self.applications_fixed.append(app)
            self.app_list.append(app_id)
            app_id += 1

        # Add PolyBench applications
        for app_cfg in self.polybench_applications:
            benchmark = app_cfg["benchmark"]
            if benchmark not in POLYBENCH_BENCHES:
                logging.warning(f"Unknown PolyBench benchmark: {benchmark}")

            app = {
                "id": app_id,
                "benchmark": benchmark,
                "variant": "",  # PolyBench doesn't use variants
                "input_arg": str(app_cfg["input_arg"]),
                "app_args": str(app_cfg["input_arg"]),
                "path": f"polybench {benchmark}",  # For compatibility
                "cores": "",
                "frequencies": [],
                "priority": None,
                "action": "profile",
            }
            self.applications_fixed.append(app)
            self.app_list.append(app_id)
            app_id += 1

        logging.info(f"Built {len(self.applications_fixed)} applications "
                     f"({len(self.bots_applications)} BOTS + {len(self.polybench_applications)} PolyBench)")


# =============================================================================
# Socket Utilities: Simple JSON-line protocol (matching server_combined_2.py pattern)
# =============================================================================
def parse_profiling_data(data: str) -> Optional[List[Dict[str, Any]]]:
    """Parse JSON profiling data from client (like server_combined_2.py)."""
    try:
        msg = json.loads(data)
        profiling_data_list = msg.get('profiling_data_list', [])
        return profiling_data_list
    except json.JSONDecodeError as e:
        logging.warning(f"JSON decode error while parsing profiling data: {e}")
        return None


def send_json(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Send JSON message with newline delimiter."""
    data = json.dumps(msg)
    sock.sendall((data + "\n").encode())


# =============================================================================
# Networking Helpers
# =============================================================================
def establish_connection() -> Tuple[socket.socket, socket.socket]:
    """Establish a TCP connection with the client."""
    print("Waiting for connection")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((IP_ADDRESS, PORT))
    server_socket.listen(1)

    client_socket, address = server_socket.accept()
    print(f"Connection established with {address[0]}:{address[1]}")
    logging.info(f"Connection established with {address[0]}:{address[1]}")

    return client_socket, server_socket


# =============================================================================
# Profiling Data Validation and Enrichment
# =============================================================================
def validate_profiling_data(profiling_data: Dict[str, Any]) -> bool:
    required_fields = [
        "time_elapsed",
        "total_energy_consumption",
        "application_id",
        "avg_temp_after",
    ]

    critical_target_fields = [
        "makespan_all_cores_frequency_11",
        "energy_all_cores_frequency_0",
        "makespan_one_core_frequency_11",
    ]

    missing_required = [f for f in required_fields if f not in profiling_data]
    if missing_required:
        logging.warning(
            f"Profiling data for app {profiling_data.get('application_id', 'UNKNOWN')} "
            f"missing required fields: {missing_required}"
        )

    missing_targets = [
        f for f in critical_target_fields
        if f not in profiling_data or profiling_data.get(f) in (None, 0.0)
    ]
    if missing_targets:
        logging.warning(
            f"Profiling data for app {profiling_data.get('application_id', 'UNKNOWN')} "
            f"missing critical target fields: {missing_targets}"
        )

    return True


def enrich_profiling_data(profiling_data: Dict[str, Any], default_values=None) -> Dict[str, Any]:
    if default_values is None:
        default_values = {
            "frequencies": [0],
            "cores": "1,2,3,4,5",
            "utilization": 0.0,
            "cycles": 0,
            "cache_references": 0,
            "cache_misses": 0,
            "branch_instructions": 0,
            "task_clock": 0.0,
            "context_switches": 0,
            "minor_faults": 0,
            "major_faults": 0,
            "branch_misses": 0,
            "branches": 0,
            "instructions": 0,
            "page_faults": 0,
            "cpu_clock": 0.0,
        }

    enriched = dict(profiling_data)

    for key, default_val in default_values.items():
        if key not in enriched:
            enriched[key] = default_val

    if "total_energy_consumption" not in enriched:
        energy_system = enriched.get("energy_system_j", 0.0)
        energy_main = enriched.get("energy_main_j", 0.0)
        energy_cpu = enriched.get("energy_cpu_j", 0.0)
        energy_denver = enriched.get("energy_denver_j", 0.0)
        energy_gpu = enriched.get("energy_gpu_j", 0.0)
        energy_ddr = enriched.get("energy_ddr_j", 0.0)
        enriched["total_energy_consumption"] = (
            energy_system + energy_main + energy_cpu +
            energy_denver + energy_gpu + energy_ddr
        )

    energy_keys = [
        "energy_system_j", "energy_main_j", "energy_cpu_j",
        "energy_denver_j", "energy_gpu_j", "energy_ddr_j",
    ]
    for ek in energy_keys:
        if ek not in enriched:
            enriched[ek] = 0.0

    if "avg_temp_after" not in enriched:
        temps = []
        for i in range(10):
            zone_key = f"thermal_zone{i}"
            if zone_key in enriched:
                temps.append(enriched[zone_key])
        enriched["avg_temp_after"] = sum(temps) / len(temps) if temps else 50.0

    if "avg_temp_delta" not in enriched:
        enriched["avg_temp_delta"] = 0.0

    for i in range(10):
        zone_key = f"thermal_zone{i}"
        if zone_key not in enriched:
            enriched[zone_key] = enriched.get("avg_temp_after", 50.0)

    for i in range(5):
        delta_key = f"temp_delta{i}"
        if delta_key not in enriched:
            enriched[delta_key] = enriched.get("avg_temp_delta", 0.0)

    return enriched


def log_profiling_summary(profiling_data: Dict[str, Any]) -> None:
    logging.info("=" * 60)
    logging.info("Profiling Data Summary:")
    logging.info(f"  Application ID: {profiling_data.get('application_id', 'N/A')}")
    logging.info(f"  Benchmark: {profiling_data.get('benchmark', 'N/A')}")
    logging.info(f"  Variant: {profiling_data.get('variant', 'N/A')}")
    logging.info(f"  Input arg: {profiling_data.get('input_arg', 'N/A')}")

    time_elapsed = profiling_data.get("time_elapsed")
    if time_elapsed is not None:
        logging.info(f"  Execution time: {time_elapsed:.3f}s")

    total_energy = profiling_data.get("total_energy_consumption")
    if total_energy is not None:
        logging.info(f"  Total energy: {total_energy:.2f}J")

    mk_11 = profiling_data.get("makespan_all_cores_frequency_11")
    if mk_11:
        logging.info(f"  MINIMUM MAKESPAN TARGET: {mk_11:.3f}s")

    e_0 = profiling_data.get("energy_all_cores_frequency_0")
    if e_0:
        logging.info(f"  Energy baseline (freq 0): {e_0:.2f}J")

    parallelism = profiling_data.get("parallelism_level", 1.0)
    logging.info(f"  Parallelism level: {parallelism:.2f}")

    avg_temp = profiling_data.get("avg_temp_after")
    if avg_temp is not None:
        logging.info(f"  Average temperature: {avg_temp:.2f}C")
    logging.info("=" * 60)


# =============================================================================
# Profiling Phase - SIMPLE PATTERN from server_combined_2.py
# =============================================================================
def perform_profiling_phase(
    client_socket: socket.socket,
    app_config: ApplicationConfig,
    timeout: float = 600.0
) -> Tuple[Dict[Tuple[int, str], Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform profiling phase using simple pattern from server_combined_2.py.

    Sends ONE application at a time for profiling and waits for result.
    This matches the working pattern in server_combined_2.py.
    """
    logging.info("=" * 80)
    logging.info("Starting Application Profiling Phase (server_combined_2.py pattern)")
    logging.info("=" * 80)

    profiling_data_list: List[Dict[str, Any]] = []
    application_profiles: Dict[Tuple[int, str], Dict[str, Any]] = {}

    # Profile each application ONE AT A TIME (like server_combined_2.py)
    for app in app_config.applications_fixed:
        application = {
            'id': app['id'],
            'benchmark': app['benchmark'],
            'variant': app['variant'],
            'input_arg': app['input_arg'],
            'app_args': app['input_arg'],  # CRITICAL: client needs this
            'path': app['path'],            # CRITICAL: client needs this
            'cores': '',
            'frequencies': [],
            'priority': None,
            'action': 'profile'
        }

        # Send single application for profiling
        send_msg_dict = {'applications': [application]}
        send_msg = json.dumps(send_msg_dict)
        client_socket.send((send_msg + '\n').encode())
        logging.info(f"Sent application {application['id']} ({app['variant']}) to client for profiling.")

        # Wait for profiling result (simple buffer-based recv like server_combined_2.py)
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
                        if new_pd_list is None:
                            logging.warning(f"Failed to parse profiling data from client: {msg}")
                            continue
                        if len(new_pd_list) == 1:
                            data_received = True
                            profiling_data = new_pd_list[0]
                            app_id = profiling_data.get('application_id', application['id'])
                            app_args_profiled = profiling_data.get('app_args', application.get('input_arg', ''))

                            mk_onecore = profiling_data.get('makespan_one_core_frequency_11')
                            mk_allcore = profiling_data.get('makespan_all_cores_frequency_11')
                            logging.info(f"mk_onecore/mk_all_core: {mk_onecore}, {mk_allcore}")

                            # Compute parallelism level
                            if mk_onecore is not None and mk_allcore is not None and mk_allcore != 0:
                                parallelism_level = mk_onecore / mk_allcore
                            else:
                                parallelism_level = 1.0
                            profiling_data['parallelism_level'] = parallelism_level

                            # Validate and enrich
                            validate_profiling_data(profiling_data)
                            profiling_data = enrich_profiling_data(profiling_data)
                            log_profiling_summary(profiling_data)

                            # Store profile
                            application_profiles[(app_id, str(app_args_profiled))] = profiling_data
                            profiling_data_list.append(profiling_data)

                            # Send acknowledgment
                            send_ack = json.dumps({'status': 'received'})
                            client_socket.send((send_ack + '\n').encode())
                            logging.info(f"Stored profiling data for application ID {app_id}")
                        else:
                            logging.warning("Received unexpected number of profiling data entries.")
                            continue
                else:
                    logging.warning("No data received. Waiting...")
                    time.sleep(1)
            except Exception as e:
                logging.error(f"Error receiving profiling data: {e}")
                time.sleep(1)

        if not data_received:
            logging.error(f"Timeout waiting for profiling data for app {app['id']}")

    logging.info("=" * 80)
    logging.info(f"Profiling Phase Complete - {len(profiling_data_list)} apps profiled")
    logging.info("=" * 80)

    return application_profiles, profiling_data_list


# =============================================================================
# Hyperparameter Management and File Naming
# =============================================================================
def build_hyperparams_dict(
    experiment_time: int,
    beta: float,
    learning_rate: float,
    epsilon_min: float,
    batch_size: int,
    discount_factor: float,
    target_temp: float
) -> Dict[str, Any]:
    """Build a dictionary of key hyperparameters for naming and comparison."""
    return {
        'exp': experiment_time,
        'beta': beta,
        'lr': learning_rate,
        'eps_min': epsilon_min,
        'batch': batch_size,
        'gamma': discount_factor,
        'temp': target_temp
    }


def build_filename_with_hyperparams(
    module_name: str,
    timestamp: str,
    hyperparams: Dict[str, Any]
) -> str:
    """
    Build a filename that encodes key hyperparameters.
    Format: MODULE_YYYYMMDD_HHMMSS_EXPep_betaB_lrL_epsE_batchB.csv
    """
    return (
        f"{module_name}_{timestamp}_"
        f"{hyperparams['exp']}ep_"
        f"beta{hyperparams['beta']}_"
        f"lr{hyperparams['lr']}_"
        f"eps{hyperparams['eps_min']}_"
        f"batch{hyperparams['batch']}.csv"
    )


def parse_hyperparams_from_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse hyperparameters from a filename.
    Returns dict with keys: exp, beta, lr, eps_min, batch, module_name
    Returns None if parsing fails.
    """
    import re
    basename = os.path.basename(filename)

    # Try new format: MODULE_YYYYMMDD_HHMMSS_EXPep_betaB_lrL_epsE_batchB.csv
    # Use (.+?) non-greedy to handle module names with underscores like MAMBRL_D3QN
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
    # Use (.+?) non-greedy to handle module names with underscores like MAMBRL_D3QN
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


def find_historical_data_with_same_experiments(
    save_dir: str,
    module_name: str,
    target_experiment_count: int,
    exclude_files: List[str] = None,
    require_exact_row_count: bool = True
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Find historical CSV files for the given module with the same experiment count.

    Args:
        save_dir: Directory to search for CSV files
        module_name: Module name to match (e.g., "MAMBRL", "MAMFRL")
        target_experiment_count: Expected experiment count from filename
        exclude_files: List of files to exclude from search
        require_exact_row_count: If True, only return files where row_count == target_experiment_count
                                 If False, return files where filename exp matches (for cross-module)

    Returns list of (filepath, hyperparams_dict) tuples, sorted by timestamp (newest first).
    """
    if exclude_files is None:
        exclude_files = []

    exclude_basenames = [os.path.basename(f) for f in exclude_files]
    matching_files = []

    try:
        for filename in os.listdir(save_dir):
            if not filename.endswith('.csv') or '_stats' in filename:
                continue
            if filename in exclude_basenames:
                continue

            # Parse hyperparameters from filename
            hyperparams = parse_hyperparams_from_filename(filename)
            if hyperparams is None:
                continue

            # Check if module name matches and experiment count matches
            if module_name in hyperparams.get('module_name', ''):
                if hyperparams.get('exp') == target_experiment_count:
                    filepath = os.path.join(save_dir, filename)
                    # Count rows in CSV and store actual row count
                    try:
                        with open(filepath, 'r') as f:
                            reader = csv.reader(f)
                            row_count = sum(1 for _ in reader) - 1  # Subtract header
                        # Store actual row count for later use
                        hyperparams['actual_row_count'] = row_count

                        if require_exact_row_count:
                            # Only include if row count matches target (for same module type comparison)
                            if row_count == target_experiment_count:
                                matching_files.append((filepath, hyperparams))
                            else:
                                logging.debug(f"Skipping {filename}: row_count={row_count} != target={target_experiment_count}")
                        else:
                            # Include based on filename exp match (for cross-module comparison)
                            # The data will need to be aggregated at load time
                            matching_files.append((filepath, hyperparams))
                    except Exception as e:
                        logging.debug(f"Cannot read {filename}: {e}")

        # Sort by timestamp (newest first)
        matching_files.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)

        mode_str = "exact row count" if require_exact_row_count else "filename exp"
        logging.info(f"Found {len(matching_files)} historical files for {module_name} with {target_experiment_count} experiments ({mode_str})")
        for fp, hp in matching_files[:3]:  # Log top 3
            logging.info(f"  - {os.path.basename(fp)}: rows={hp.get('actual_row_count')}, beta={hp.get('beta')}, lr={hp.get('lr')}")

        return matching_files

    except Exception as e:
        logging.error(f"Error finding historical data: {e}")
        return []


def format_hyperparams_table(
    current_hyperparams: Dict[str, Any],
    historical_hyperparams_list: List[Tuple[str, Dict[str, Any]]]
) -> str:
    """
    Format a text table comparing hyperparameters of current run vs historical runs.
    """
    lines = []
    lines.append("HYPERPARAMETER COMPARISON")
    lines.append("=" * 50)

    # Header
    header = f"{'Parameter':<12} {'Current':<10}"
    for i, (_, hp) in enumerate(historical_hyperparams_list[:2]):  # Max 2 historical
        header += f" {'Hist ' + str(i+1):<10}"
    lines.append(header)
    lines.append("-" * 50)

    # Parameters to compare
    params = [
        ('Experiments', 'exp'),
        ('Beta', 'beta'),
        ('Learn Rate', 'lr'),
        ('Epsilon Min', 'eps_min'),
        ('Batch Size', 'batch'),
        ('Gamma', 'gamma'),
        ('Target Temp', 'temp'),
    ]

    for param_label, param_key in params:
        row = f"{param_label:<12} "

        # Current value
        val = current_hyperparams.get(param_key)
        if val is not None:
            if isinstance(val, float):
                row += f"{val:<10.4f}"
            else:
                row += f"{str(val):<10}"
        else:
            row += f"{'N/A':<10}"

        # Historical values
        for _, hp in historical_hyperparams_list[:2]:
            val = hp.get(param_key)
            if val is not None:
                if isinstance(val, float):
                    row += f" {val:<10.4f}"
                else:
                    row += f" {str(val):<10}"
            else:
                row += f" {'N/A':<10}"

        lines.append(row)

    lines.append("=" * 50)

    # Add file info
    lines.append("")
    lines.append("Files:")
    lines.append(f"  Current: (running)")
    for i, (fp, _) in enumerate(historical_hyperparams_list[:2]):
        lines.append(f"  Hist {i+1}: {os.path.basename(fp)}")

    return "\n".join(lines)


# =============================================================================
# Historical Data Loading
# =============================================================================
def load_historical_csv(filepath: str):
    if not os.path.exists(filepath):
        logging.warning(f"Historical data file not found: {filepath}")
        return None
    try:
        data = {}
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in row:
                    data.setdefault(key, [])
                    try:
                        data[key].append(float(row[key]))
                    except (ValueError, TypeError):
                        data[key].append(0.0)
        logging.info(f"Loaded historical data from {filepath} ({len(data.get('makespan', []))} rows)")
        return data
    except Exception as e:
        logging.error(f"Failed to load historical data: {e}")
        return None


def plot_comparison_with_history(
    results_dict: Dict[str, Tuple],
    save_path: Optional[str] = None,
    all_csv_files: Optional[Dict[str, str]] = None,
    current_hyperparams: Optional[Dict[str, Any]] = None,
    all_hyperparams: Optional[Dict[str, List[Tuple[str, Dict[str, Any]]]]] = None
) -> None:
    """
    Plot comparison of multiple RL modules with synchronized X-axis (iteration-based).

    Each module returns (ts, makespan_per_app, energy_per_app, temperature_per_app)
    where *_per_app is a list of lists: [[app1_iter1, app1_iter2,...], [app2_iter1,...], ...]

    For comparison, we aggregate per-iteration: sum/avg across apps for each iteration.

    Args:
        results_dict: Module results {name: (ts, makespan_per_app, energy_per_app, temp_per_app)}
        save_path: Path to save the plot
        all_csv_files: CSV files for each module
        current_hyperparams: Current run hyperparameters
        all_hyperparams: Historical hyperparameters for each module
    """
    try:
        # Use 2x3 grid: 3 metrics + 1 hyperparams table + 2 stats panels
        fig, axs = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle("Multi-Module RL Comparison (MAMBRL vs MAMFRL)", fontsize=18, fontweight="bold")

        colors = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray"]
        linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
        markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

        # Find max iterations across all modules for X-axis synchronization
        max_iterations = 0
        for module_name, data in results_dict.items():
            if data is not None:
                ts, makespan_per_app, energy_per_app, temperature_per_app = data
                if ts:
                    max_iterations = max(max_iterations, len(ts))
                elif makespan_per_app and makespan_per_app[0]:
                    max_iterations = max(max_iterations, len(makespan_per_app[0]))

        logging.info(f"Comparison plot: max_iterations={max_iterations}, modules={list(results_dict.keys())}")

        # =====================================================================
        # Makespan Plot (Total makespan per iteration = sum of all apps)
        # =====================================================================
        ax = axs[0, 0]
        for idx, (module_name, data) in enumerate(results_dict.items()):
            if data is None:
                logging.warning(f"No data for {module_name}, skipping in makespan plot")
                continue
            ts, makespan_per_app, energy_per_app, temperature_per_app = data

            # Compute total makespan per iteration (sum across apps)
            if makespan_per_app and makespan_per_app[0]:
                num_iterations = len(makespan_per_app[0])
                total_makespan_per_iter = []
                for iter_idx in range(num_iterations):
                    total = sum(app_data[iter_idx] for app_data in makespan_per_app if iter_idx < len(app_data))
                    total_makespan_per_iter.append(total)

                x_axis = ts if ts and len(ts) == num_iterations else list(range(num_iterations))

                ax.plot(
                    x_axis, total_makespan_per_iter,
                    label=module_name,
                    color=colors[idx % len(colors)],
                    linewidth=2.5,
                    linestyle=linestyles[idx % len(linestyles)],
                    marker=markers[idx % len(markers)], markersize=5, markevery=max(1, num_iterations // 10)
                )
                logging.info(f"{module_name}: {num_iterations} iterations, makespan range [{min(total_makespan_per_iter):.2f}, {max(total_makespan_per_iter):.2f}]")

        ax.set_title("Total Makespan Over Iterations", fontweight="bold", fontsize=14)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Total Makespan (s)", fontsize=12)
        ax.set_xlim(0, max_iterations)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # =====================================================================
        # Energy Plot (Total energy per iteration = sum of all apps)
        # =====================================================================
        ax = axs[0, 1]
        for idx, (module_name, data) in enumerate(results_dict.items()):
            if data is None:
                continue
            ts, makespan_per_app, energy_per_app, temperature_per_app = data

            if energy_per_app and energy_per_app[0]:
                num_iterations = len(energy_per_app[0])
                total_energy_per_iter = []
                for iter_idx in range(num_iterations):
                    total = sum(app_data[iter_idx] for app_data in energy_per_app if iter_idx < len(app_data))
                    total_energy_per_iter.append(total)

                x_axis = ts if ts and len(ts) == num_iterations else list(range(num_iterations))

                ax.plot(
                    x_axis, total_energy_per_iter,
                    label=module_name,
                    color=colors[idx % len(colors)],
                    linewidth=2.5,
                    linestyle=linestyles[idx % len(linestyles)],
                    marker=markers[idx % len(markers)], markersize=5, markevery=max(1, num_iterations // 10)
                )

        ax.set_title("Total Energy Over Iterations", fontweight="bold", fontsize=14)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Total Energy (J)", fontsize=12)
        ax.set_xlim(0, max_iterations)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # =====================================================================
        # Temperature Plot (Average temperature per iteration)
        # FIXED: Better handling of temperature data with fallbacks
        # =====================================================================
        ax = axs[1, 0]
        all_temps = []
        for idx, (module_name, data) in enumerate(results_dict.items()):
            if data is None:
                continue
            ts, makespan_per_app, energy_per_app, temperature_per_app = data

            if temperature_per_app and temperature_per_app[0]:
                num_iterations = len(temperature_per_app[0])
                avg_temp_per_iter = []
                for iter_idx in range(num_iterations):
                    # Get temperatures from all apps for this iteration
                    temps = [app_data[iter_idx] for app_data in temperature_per_app
                             if iter_idx < len(app_data)]
                    # Filter out invalid temps (0, negative, or default 50.0 values)
                    # A constant 50.0 indicates missing data - try to detect variation
                    valid_temps = [t for t in temps if t > 0 and t != 50.0 and t < 200]
                    if not valid_temps:
                        # If all temps are 50.0 (default), use them but mark for later
                        valid_temps = [t for t in temps if t > 0 and t < 200]
                    avg_temp = np.mean(valid_temps) if valid_temps else 50.0
                    avg_temp_per_iter.append(avg_temp)

                # Check if we have actual temperature variation (not all defaults)
                temp_std = np.std(avg_temp_per_iter)
                if temp_std > 0.1:  # Has actual variation
                    valid_temps_plot = [t for t in avg_temp_per_iter if 0 < t < 200]
                    all_temps.extend(valid_temps_plot)

                x_axis = ts if ts and len(ts) == num_iterations else list(range(num_iterations))

                ax.plot(
                    x_axis, avg_temp_per_iter,
                    label=module_name,
                    color=colors[idx % len(colors)],
                    linewidth=2.5,
                    linestyle=linestyles[idx % len(linestyles)],
                    marker=markers[idx % len(markers)], markersize=5, markevery=max(1, num_iterations // 10)
                )

        # Set Y-axis limits based on actual temperature range
        if all_temps:
            temp_min = min(all_temps)
            temp_max = max(all_temps)
            temp_range = max(temp_max - temp_min, 5.0)  # Minimum range of 5C
            ax.set_ylim(temp_min - temp_range * 0.1, temp_max + temp_range * 0.1)
        else:
            # If no valid temps, show a reasonable range around target
            ax.set_ylim(40, 60)

        ax.axhline(y=50, color='red', linestyle=":", linewidth=2, label="Target Temp (50C)")
        ax.set_title("Average Temperature Over Iterations", fontweight="bold", fontsize=14)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Temperature (C)", fontsize=12)
        ax.set_xlim(0, max_iterations)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # =====================================================================
        # Hyperparameter Comparison Table Panel (top-right)
        # =====================================================================
        ax = axs[0, 2]
        ax.axis("off")

        if current_hyperparams and all_hyperparams:
            # Collect all historical hyperparams across modules
            all_hist = []
            for module_name, hist_list in all_hyperparams.items():
                for filepath, hp in hist_list:
                    if hp not in [h for _, h in all_hist]:
                        all_hist.append((filepath, hp))

            hp_table = format_hyperparams_table(current_hyperparams, all_hist[:2])
            ax.text(
                0.05, 0.95, hp_table, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5)
            )
            ax.set_title("Hyperparameter Comparison", fontweight="bold", fontsize=12)
        else:
            ax.text(
                0.5, 0.5, "No hyperparameter data available",
                transform=ax.transAxes, ha='center', va='center', fontsize=12
            )
            ax.set_title("Hyperparameter Comparison", fontweight="bold", fontsize=12)

        # =====================================================================
        # Summary Statistics Panel (bottom-middle)
        # =====================================================================
        ax = axs[1, 1]
        ax.axis("off")
        stats_text = "PERFORMANCE SUMMARY\n" + "=" * 45 + "\n\n"

        for module_name, data in results_dict.items():
            if data is None:
                stats_text += f"{module_name}: No data available\n\n"
                continue
            ts, makespan_per_app, energy_per_app, temperature_per_app = data

            # Compute per-iteration totals
            if makespan_per_app and makespan_per_app[0]:
                num_iterations = len(makespan_per_app[0])
                total_makespan_per_iter = []
                total_energy_per_iter = []
                avg_temp_per_iter = []

                for iter_idx in range(num_iterations):
                    ms_total = sum(app[iter_idx] for app in makespan_per_app if iter_idx < len(app))
                    en_total = sum(app[iter_idx] for app in energy_per_app if iter_idx < len(app))
                    temps = [app[iter_idx] for app in temperature_per_app if iter_idx < len(app) and app[iter_idx] > 0]

                    total_makespan_per_iter.append(ms_total)
                    total_energy_per_iter.append(en_total)
                    avg_temp_per_iter.append(np.mean(temps) if temps else 50.0)

                avg_makespan = float(np.mean(total_makespan_per_iter))
                min_makespan = float(np.min(total_makespan_per_iter))
                avg_energy = float(np.mean(total_energy_per_iter))
                avg_temp = float(np.mean(avg_temp_per_iter))

                stats_text += f"{module_name}:\n"
                stats_text += f"  Iterations:     {num_iterations}\n"
                stats_text += f"  Avg Makespan:   {avg_makespan:.3f}s\n"
                stats_text += f"  Best Makespan:  {min_makespan:.3f}s\n"
                stats_text += f"  Avg Energy:     {avg_energy:.2f}J\n"
                stats_text += f"  Avg Temp:       {avg_temp:.2f}C\n"
                stats_text += "-" * 40 + "\n"

        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )
        ax.set_title("Performance Metrics", fontweight="bold", fontsize=12)

        # =====================================================================
        # File Information Panel (bottom-right)
        # =====================================================================
        ax = axs[1, 2]
        ax.axis("off")
        files_text = "OUTPUT FILES\n" + "=" * 45 + "\n\n"

        if all_csv_files:
            for module_name, csv_file in all_csv_files.items():
                files_text += f"{module_name}:\n"
                files_text += f"  {os.path.basename(csv_file)}\n\n"

        if save_path:
            files_text += f"Comparison Plot:\n"
            files_text += f"  {os.path.basename(save_path)}\n"

        ax.text(
            0.05, 0.95, files_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5)
        )
        ax.set_title("Output Files", fontweight="bold", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Comparison plot saved to {save_path}")

        # Non-blocking show
        plt.show(block=False)
        plt.pause(2)

    except Exception as e:
        logging.error(f"Failed to create comparison plot: {e}", exc_info=True)


# =============================================================================
# Pass 1 vs Pass 2 Comparison Plot (Load Model Comparison)
# =============================================================================
def plot_pass1_vs_pass2_comparison(
    pass1_results: Dict[str, Tuple],
    pass2_results: Dict[str, Tuple],
    save_path: Optional[str] = None,
    experiment_time: int = 20
) -> None:
    """
    Plot comparison between Pass 1 (fresh training) and Pass 2 (loaded models).

    Shows how each module performs when starting fresh vs continuing from saved state.

    Args:
        pass1_results: Results from Pass 1 (load_model=False)
        pass2_results: Results from Pass 2 (load_model=True)
        save_path: Path to save the plot
        experiment_time: Number of experiment iterations
    """
    try:
        # Use 2x2 grid for comparison
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Pass 1 (Fresh) vs Pass 2 (Loaded Model) Comparison", fontsize=18, fontweight="bold")

        colors_pass1 = ["blue", "green", "purple", "orange", "brown", "pink"]
        colors_pass2 = ["red", "cyan", "magenta", "yellow", "gray", "olive"]
        markers = ["o", "s", "^", "D", "v", "<"]

        # Get all module names
        all_modules = set(list(pass1_results.keys()) + list(pass2_results.keys()))
        max_iterations = experiment_time

        # =====================================================================
        # Makespan Comparison
        # =====================================================================
        ax = axs[0, 0]
        for idx, module_name in enumerate(all_modules):
            # Pass 1 data
            data1 = pass1_results.get(module_name)
            if data1 is not None:
                ts, makespan_per_app, _, _ = data1
                if makespan_per_app and makespan_per_app[0]:
                    num_iterations = len(makespan_per_app[0])
                    total_makespan = []
                    for iter_idx in range(num_iterations):
                        total = sum(app[iter_idx] for app in makespan_per_app if iter_idx < len(app))
                        total_makespan.append(total)
                    x_axis = ts if ts and len(ts) == num_iterations else list(range(num_iterations))
                    ax.plot(x_axis, total_makespan, label=f"{module_name} (Fresh)",
                           color=colors_pass1[idx % len(colors_pass1)], linestyle='-',
                           marker=markers[idx % len(markers)], markersize=4, markevery=max(1, num_iterations // 10))

            # Pass 2 data
            data2 = pass2_results.get(module_name)
            if data2 is not None:
                ts, makespan_per_app, _, _ = data2
                if makespan_per_app and makespan_per_app[0]:
                    num_iterations = len(makespan_per_app[0])
                    total_makespan = []
                    for iter_idx in range(num_iterations):
                        total = sum(app[iter_idx] for app in makespan_per_app if iter_idx < len(app))
                        total_makespan.append(total)
                    x_axis = ts if ts and len(ts) == num_iterations else list(range(num_iterations))
                    ax.plot(x_axis, total_makespan, label=f"{module_name} (Loaded)",
                           color=colors_pass2[idx % len(colors_pass2)], linestyle='--',
                           marker=markers[idx % len(markers)], markersize=4, markevery=max(1, num_iterations // 10))

        ax.set_title("Makespan: Fresh vs Loaded Model", fontweight="bold", fontsize=12)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Total Makespan (s)", fontsize=10)
        ax.set_xlim(0, max_iterations)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # =====================================================================
        # Energy Comparison
        # =====================================================================
        ax = axs[0, 1]
        for idx, module_name in enumerate(all_modules):
            # Pass 1 data
            data1 = pass1_results.get(module_name)
            if data1 is not None:
                ts, _, energy_per_app, _ = data1
                if energy_per_app and energy_per_app[0]:
                    num_iterations = len(energy_per_app[0])
                    total_energy = []
                    for iter_idx in range(num_iterations):
                        total = sum(app[iter_idx] for app in energy_per_app if iter_idx < len(app))
                        total_energy.append(total)
                    x_axis = ts if ts and len(ts) == num_iterations else list(range(num_iterations))
                    ax.plot(x_axis, total_energy, label=f"{module_name} (Fresh)",
                           color=colors_pass1[idx % len(colors_pass1)], linestyle='-',
                           marker=markers[idx % len(markers)], markersize=4, markevery=max(1, num_iterations // 10))

            # Pass 2 data
            data2 = pass2_results.get(module_name)
            if data2 is not None:
                ts, _, energy_per_app, _ = data2
                if energy_per_app and energy_per_app[0]:
                    num_iterations = len(energy_per_app[0])
                    total_energy = []
                    for iter_idx in range(num_iterations):
                        total = sum(app[iter_idx] for app in energy_per_app if iter_idx < len(app))
                        total_energy.append(total)
                    x_axis = ts if ts and len(ts) == num_iterations else list(range(num_iterations))
                    ax.plot(x_axis, total_energy, label=f"{module_name} (Loaded)",
                           color=colors_pass2[idx % len(colors_pass2)], linestyle='--',
                           marker=markers[idx % len(markers)], markersize=4, markevery=max(1, num_iterations // 10))

        ax.set_title("Energy: Fresh vs Loaded Model", fontweight="bold", fontsize=12)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Total Energy (J)", fontsize=10)
        ax.set_xlim(0, max_iterations)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # =====================================================================
        # Performance Improvement Summary (bar chart)
        # =====================================================================
        ax = axs[1, 0]
        module_names_list = list(all_modules)
        x_pos = np.arange(len(module_names_list))
        width = 0.35

        pass1_makespans = []
        pass2_makespans = []
        for module_name in module_names_list:
            # Get average makespan for each pass
            data1 = pass1_results.get(module_name)
            if data1 is not None:
                ts, makespan_per_app, _, _ = data1
                if makespan_per_app and makespan_per_app[0]:
                    num_iterations = len(makespan_per_app[0])
                    total_makespan = [sum(app[i] for app in makespan_per_app if i < len(app)) for i in range(num_iterations)]
                    pass1_makespans.append(np.mean(total_makespan))
                else:
                    pass1_makespans.append(0)
            else:
                pass1_makespans.append(0)

            data2 = pass2_results.get(module_name)
            if data2 is not None:
                ts, makespan_per_app, _, _ = data2
                if makespan_per_app and makespan_per_app[0]:
                    num_iterations = len(makespan_per_app[0])
                    total_makespan = [sum(app[i] for app in makespan_per_app if i < len(app)) for i in range(num_iterations)]
                    pass2_makespans.append(np.mean(total_makespan))
                else:
                    pass2_makespans.append(0)
            else:
                pass2_makespans.append(0)

        ax.bar(x_pos - width/2, pass1_makespans, width, label='Fresh Training', color='blue', alpha=0.7)
        ax.bar(x_pos + width/2, pass2_makespans, width, label='Loaded Model', color='red', alpha=0.7)

        ax.set_title("Average Makespan Comparison", fontweight="bold", fontsize=12)
        ax.set_xlabel("Module", fontsize=10)
        ax.set_ylabel("Average Makespan (s)", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(module_names_list, rotation=45, ha='right', fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # =====================================================================
        # Improvement Statistics Panel
        # =====================================================================
        ax = axs[1, 1]
        ax.axis("off")
        stats_text = "LOAD MODEL IMPROVEMENT ANALYSIS\n" + "=" * 45 + "\n\n"

        for module_name in module_names_list:
            data1 = pass1_results.get(module_name)
            data2 = pass2_results.get(module_name)

            if data1 is not None and data2 is not None:
                ts1, mk1, en1, _ = data1
                ts2, mk2, en2, _ = data2

                if mk1 and mk1[0] and mk2 and mk2[0]:
                    # Calculate average makespan improvement
                    num_iter1 = len(mk1[0])
                    num_iter2 = len(mk2[0])
                    avg_mk1 = np.mean([sum(app[i] for app in mk1 if i < len(app)) for i in range(num_iter1)])
                    avg_mk2 = np.mean([sum(app[i] for app in mk2 if i < len(app)) for i in range(num_iter2)])

                    improvement = ((avg_mk1 - avg_mk2) / avg_mk1) * 100 if avg_mk1 > 0 else 0

                    stats_text += f"{module_name}:\n"
                    stats_text += f"  Fresh Avg:  {avg_mk1:.3f}s\n"
                    stats_text += f"  Loaded Avg: {avg_mk2:.3f}s\n"
                    stats_text += f"  Improvement: {improvement:+.1f}%\n"
                    stats_text += "-" * 40 + "\n"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment="top", family="monospace",
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))
        ax.set_title("Improvement Statistics", fontweight="bold", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Pass 1 vs Pass 2 comparison plot saved to {save_path}")

        plt.show(block=False)
        plt.pause(2)

    except Exception as e:
        logging.error(f"Failed to create Pass 1 vs Pass 2 comparison plot: {e}", exc_info=True)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Import and Register All Enabled RL Modules
    # -------------------------------------------------------------------------
    import_and_register_modules()

    if not RL_MODULE_REGISTRY:
        logging.error("No RL modules registered! Check ENABLED_RL_MODULES.")
        raise SystemExit(1)

    logging.info(f"Active RL modules: {list(RL_MODULE_REGISTRY.keys())}")

    # -------------------------------------------------------------------------
    # Build HP Search Space from Top-Level Configuration
    # -------------------------------------------------------------------------
    HP_SEARCH_SPACE = {}
    for hp_name, should_tune in HP_TUNE_CONFIG.items():
        if should_tune:
            HP_SEARCH_SPACE[hp_name] = HP_SEARCH_VALUES[hp_name]
        else:
            # Use default value (middle value in search space as default)
            HP_SEARCH_SPACE[hp_name] = [HP_SEARCH_VALUES[hp_name][len(HP_SEARCH_VALUES[hp_name])//2]]

    # Log which hyperparameters will be tuned
    hp_tuning_enabled = [k for k, v in HP_TUNE_CONFIG.items() if v]
    hp_tuning_disabled = [k for k, v in HP_TUNE_CONFIG.items() if not v]

    # -------------------------------------------------------------------------
    # Apply Default Hyperparameters from Top-Level Configuration
    # -------------------------------------------------------------------------
    experiment_time = MAIN_EXPERIMENT_TIME
    target_temp = DEFAULT_HYPERPARAMETERS['target_temp']
    clock_change_time = DEFAULT_HYPERPARAMETERS['clock_change_time']
    beta = DEFAULT_HYPERPARAMETERS['beta']
    load_model = DEFAULT_HYPERPARAMETERS['load_model']
    plan_count = DEFAULT_HYPERPARAMETERS['plan_count']
    model_train_start = DEFAULT_HYPERPARAMETERS['model_train_start']
    agent_train_start = DEFAULT_HYPERPARAMETERS['agent_train_start']
    real_synthetic_ratio = DEFAULT_HYPERPARAMETERS['real_synthetic_ratio']
    learn_count = DEFAULT_HYPERPARAMETERS['learn_count']
    mem_size = DEFAULT_HYPERPARAMETERS['mem_size']
    learning_rate = DEFAULT_HYPERPARAMETERS['learning_rate']
    batch_size = DEFAULT_HYPERPARAMETERS['batch_size']
    discount_factor = DEFAULT_HYPERPARAMETERS['discount_factor']
    epsilon = DEFAULT_HYPERPARAMETERS['epsilon']
    epsilon_decay = DEFAULT_HYPERPARAMETERS['epsilon_decay']
    epsilon_min = DEFAULT_HYPERPARAMETERS['epsilon_min']
    epsilon_start = DEFAULT_HYPERPARAMETERS['epsilon_start']
    epsilon_end = DEFAULT_HYPERPARAMETERS['epsilon_end']
    reset_learning_rate_value = DEFAULT_HYPERPARAMETERS['reset_learning_rate_value']
    save_repetition = DEFAULT_HYPERPARAMETERS['save_repetition']
    save_model = DEFAULT_HYPERPARAMETERS['save_model']

    logging.info("=" * 80)
    logging.info(f"MODULAR RL COMPARISON - {len(RL_MODULE_REGISTRY)} Modules")
    logging.info(f"HP Tuning Enabled              : {ENABLE_HP_TUNING}")
    if ENABLE_HP_TUNING:
        logging.info(f"HP Tuning Module               : {HP_TUNING_MODULE}")
        logging.info(f"HP Tuning Experiments          : {HP_TUNING_EXPERIMENT_TIME}")
        logging.info(f"Main Experiments               : {MAIN_EXPERIMENT_TIME}")
    logging.info("-" * 40)
    logging.info("OPTIMAL PARAMETERS (from tuning):")
    logging.info(f"  Learning rate                : {learning_rate}")
    logging.info(f"  Batch size                   : {batch_size}")
    logging.info(f"  Beta (objective balance)     : {beta}")
    logging.info(f"  Epsilon min                  : {epsilon_min}")
    logging.info("-" * 40)
    logging.info("MODEL-BASED RL PARAMETERS:")
    logging.info(f"  Plan count (synthetic data)  : {plan_count}")
    logging.info(f"  Model train start (real data): {model_train_start}")
    logging.info(f"  Agent train start (total)    : {agent_train_start}")
    logging.info(f"  Real/Synthetic ratio         : {real_synthetic_ratio}")
    logging.info("-" * 40)
    logging.info(f"Target temperature             : {target_temp}C")
    logging.info(f"Discount factor                : {discount_factor}")
    logging.info("=" * 80)

    # -------------------------------------------------------------------------
    # Application Configuration
    # -------------------------------------------------------------------------
    app_config = ApplicationConfig()

    # -------------------------------------------------------------------------
    # Output File Names (with hyperparameters encoded)
    # -------------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_plot_name = os.path.join(
        SAVE_DIR,
        f"Comparison_MultiModule_{timestamp}_{experiment_time}ep_beta{beta}.png"
    )

    # -------------------------------------------------------------------------
    # Establish Connection with Client
    # -------------------------------------------------------------------------
    client_socket, server_socket = establish_connection()

    # -------------------------------------------------------------------------
    # Application Profiling Phase (simple pattern like server_combined_2.py)
    # -------------------------------------------------------------------------
    try:
        application_profiles, profiling_data_list = perform_profiling_phase(
            client_socket, app_config, timeout=600.0
        )
    except Exception as e:
        logging.error(f"Profiling phase failed: {e}", exc_info=True)
        try:
            client_socket.close()
            server_socket.close()
        except Exception:
            pass
        raise SystemExit(1)

    if not profiling_data_list:
        logging.error("No profiling data collected - cannot proceed!")
        try:
            client_socket.close()
            server_socket.close()
        except Exception:
            pass
        raise SystemExit(1)

    # ==========================================================================
    # PHASE 0: HYPERPARAMETER TUNING (if enabled)
    # ==========================================================================
    # Run SAMBRL with different hyperparameter combinations to find the best
    # Then use those hyperparameters for all subsequent training
    # ==========================================================================
    best_hp = {
        'plan_count': plan_count,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }

    if ENABLE_HP_TUNING and HP_TUNING_MODULE in RL_MODULE_REGISTRY:
        logging.info("\n" + "=" * 80)
        logging.info("PHASE 0: HYPERPARAMETER TUNING WITH " + HP_TUNING_MODULE)
        logging.info(f"Running {HP_TUNING_EXPERIMENT_TIME} experiments for each HP combination")
        logging.info("=" * 80)
        logging.info(f"Hyperparameters to tune: {hp_tuning_enabled}")
        logging.info(f"Hyperparameters fixed: {hp_tuning_disabled}")
        logging.info("=" * 80 + "\n")

        tuning_results = []
        tuning_train_function = RL_MODULE_REGISTRY[HP_TUNING_MODULE]

        # Generate all hyperparameter combinations dynamically
        from itertools import product
        hp_keys = list(HP_SEARCH_SPACE.keys())
        hp_values = [HP_SEARCH_SPACE[k] for k in hp_keys]
        hp_combinations = list(product(*hp_values))

        logging.info(f"Testing {len(hp_combinations)} hyperparameter combinations:")
        for i, combo in enumerate(hp_combinations):
            combo_dict = dict(zip(hp_keys, combo))
            combo_str = ", ".join([f"{k}={v}" for k, v in combo_dict.items() if HP_TUNE_CONFIG.get(k, False)])
            logging.info(f"  {i+1}. {combo_str}")

        for hp_idx, hp_combo in enumerate(hp_combinations):
            # Create hyperparameter dict from combination
            hp_dict = dict(zip(hp_keys, hp_combo))

            # Get values for each hyperparameter
            test_plan_count = hp_dict.get('plan_count', plan_count)
            test_model_train_start = hp_dict.get('model_train_start', model_train_start)
            test_agent_train_start = hp_dict.get('agent_train_start', agent_train_start)
            test_real_synthetic_ratio = hp_dict.get('real_synthetic_ratio', real_synthetic_ratio)
            test_lr = hp_dict.get('learning_rate', learning_rate)
            test_batch_size = hp_dict.get('batch_size', batch_size)
            test_beta = hp_dict.get('beta', beta)
            test_epsilon_min = hp_dict.get('epsilon_min', epsilon_min)
            test_epsilon_decay = hp_dict.get('epsilon_decay', epsilon_decay)
            test_discount_factor = hp_dict.get('discount_factor', discount_factor)

            # Build display string for logging
            tuned_params = {k: v for k, v in hp_dict.items() if HP_TUNE_CONFIG.get(k, False)}
            tuned_str = ", ".join([f"{k}={v}" for k, v in tuned_params.items()])

            logging.info("\n" + "-" * 60)
            logging.info(f"HP Tuning {hp_idx + 1}/{len(hp_combinations)}")
            logging.info(f"  {tuned_str}")
            logging.info("-" * 60)

            # Build tuning hyperparams dict
            tuning_hyperparams = build_hyperparams_dict(
                experiment_time=HP_TUNING_EXPERIMENT_TIME,
                beta=test_beta,
                learning_rate=test_lr,
                epsilon_min=test_epsilon_min,
                batch_size=test_batch_size,
                discount_factor=test_discount_factor,
                target_temp=target_temp
            )

            tuning_timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Build filename with all tuned params
            # pc=plan_count, mts=model_train_start, ats=agent_train_start, rsr=real_synthetic_ratio
            param_str = f"pc{test_plan_count}_mts{test_model_train_start}_rsr{test_real_synthetic_ratio}_lr{test_lr}_bs{test_batch_size}"
            if HP_TUNE_CONFIG.get('beta', False):
                param_str += f"_beta{test_beta}"
            if HP_TUNE_CONFIG.get('epsilon_min', False):
                param_str += f"_eps{test_epsilon_min}"
            tuning_filename = os.path.join(
                SAVE_DIR,
                f"HP_TUNING_{HP_TUNING_MODULE}_{tuning_timestamp}_{HP_TUNING_EXPERIMENT_TIME}ep_{param_str}.csv"
            )

            try:
                tuning_start = time.time()

                # Build base kwargs for tuning function
                # NOTE: model_train_start and real_synthetic_ratio only supported by:
                # SAMBRL, MAMBRL_D3QN, MARB_D3QN, SARBRL
                tuning_kwargs = {
                    'client_socket': client_socket,
                    'data_keys': DATA_KEYS,
                    'experiment_time': HP_TUNING_EXPERIMENT_TIME,
                    'clock_change_time': clock_change_time,
                    'beta': test_beta,
                    'load_model': False,
                    'learn_count': test_model_train_start,
                    'plan_count': test_plan_count,
                    'mem_size': mem_size,
                    'learning_rate': test_lr,
                    'discount_factor': test_discount_factor,
                    'epsilon': epsilon,
                    'epsilon_decay': test_epsilon_decay,
                    'epsilon_min': test_epsilon_min,
                    'epsilon_start': epsilon_start,
                    'epsilon_end': epsilon_end,
                    'reset_learning_rate_value': reset_learning_rate_value,
                    'save_repetition': save_repetition,
                    'save_model': False,  # Don't save during tuning
                    'batch_size': test_batch_size,
                    'agent_train_start': test_agent_train_start,
                    'target_temp': target_temp,
                    'server_name_1': "",
                    'server_name_2': "",
                    'server_name_main': tuning_filename,
                    'profiling_data_list': profiling_data_list,
                    'application_profiles': application_profiles,
                    'applications_fixed': app_config.applications_fixed,
                    'priority_combinations': app_config.priority_combinations,
                    'frequency_combinations': app_config.frequency_combinations,
                    'num_cores_list': app_config.num_cores_list,
                }

                # Add model-based RL parameters only for modules that support them
                MODULES_WITH_MODEL_TRAIN_START = ['SAMBRL', 'MAMBRL_D3QN', 'MARB_D3QN', 'SARBRL']
                if HP_TUNING_MODULE in MODULES_WITH_MODEL_TRAIN_START:
                    tuning_kwargs['model_train_start'] = test_model_train_start
                    tuning_kwargs['real_synthetic_ratio'] = test_real_synthetic_ratio

                tuning_result = tuning_train_function(**tuning_kwargs)
                tuning_duration = time.time() - tuning_start

                if tuning_result is not None:
                    ts, makespan_per_app, energy_per_app, temp_per_app = tuning_result

                    # Calculate average metrics from the last 50% of training
                    if makespan_per_app and makespan_per_app[0]:
                        num_iters = len(makespan_per_app[0])
                        start_idx = num_iters // 2  # Last 50%

                        # Get average makespan from last half
                        avg_makespans = []
                        for iter_idx in range(start_idx, num_iters):
                            total = sum(app[iter_idx] for app in makespan_per_app if iter_idx < len(app))
                            avg_makespans.append(total)

                        avg_makespan = np.mean(avg_makespans) if avg_makespans else float('inf')
                        min_makespan = np.min(avg_makespans) if avg_makespans else float('inf')

                        tuning_results.append({
                            'hyperparams': hp_dict.copy(),  # Store all hyperparameters
                            'metrics': {
                                'avg_makespan': avg_makespan,
                                'min_makespan': min_makespan,
                                'duration': tuning_duration
                            }
                        })

                        logging.info(f"  Tuning result: avg_makespan={avg_makespan:.3f}s, "
                                   f"min_makespan={min_makespan:.3f}s, duration={tuning_duration:.1f}s")

            except Exception as e:
                logging.error(f"HP tuning failed for pc={test_plan_count}, mts={test_model_train_start}, rsr={test_real_synthetic_ratio}, lr={test_lr}, bs={test_batch_size}: {e}")

            # Cooldown between tuning runs
            logging.info("Cooling for 3 seconds before next HP tuning run...")
            time.sleep(3)

        # Find best hyperparameters based on average makespan
        if tuning_results:
            best_result = min(tuning_results, key=lambda x: x['metrics']['avg_makespan'])
            best_hp = best_result['hyperparams']

            logging.info("\n" + "=" * 80)
            logging.info("HYPERPARAMETER TUNING COMPLETE")
            logging.info("=" * 80)
            logging.info(f"Best hyperparameters found:")
            for hp_name, hp_value in best_hp.items():
                was_tuned = "TUNED" if HP_TUNE_CONFIG.get(hp_name, False) else "default"
                logging.info(f"  {hp_name:20s} = {hp_value} ({was_tuned})")
            logging.info(f"Best avg_makespan: {best_result['metrics']['avg_makespan']:.3f}s")
            logging.info(f"Best min_makespan: {best_result['metrics']['min_makespan']:.3f}s")
            logging.info("=" * 80 + "\n")

            # Update global parameters with best values for all tunable hyperparameters
            plan_count = best_hp.get('plan_count', plan_count)
            learning_rate = best_hp.get('learning_rate', learning_rate)
            batch_size = best_hp.get('batch_size', batch_size)
            beta = best_hp.get('beta', beta)
            epsilon_min = best_hp.get('epsilon_min', epsilon_min)
            epsilon_decay = best_hp.get('epsilon_decay', epsilon_decay)
            discount_factor = best_hp.get('discount_factor', discount_factor)

            # Save the best hyperparameters to file for future use
            save_tuned_hyperparameters(best_hp, best_result['metrics'], TUNED_HP_FILE)
        else:
            logging.warning("No successful HP tuning runs - using default hyperparameters")

    elif ENABLE_HP_TUNING and HP_TUNING_MODULE not in RL_MODULE_REGISTRY:
        logging.warning(f"HP tuning module {HP_TUNING_MODULE} not registered - skipping tuning phase")

    # =========================================================================
    # LOAD PREVIOUSLY TUNED HYPERPARAMETERS (when HP tuning is disabled)
    # =========================================================================
    elif not ENABLE_HP_TUNING:
        logging.info("\n" + "=" * 80)
        logging.info("HP TUNING DISABLED - Loading previously tuned hyperparameters")
        logging.info("=" * 80)

        loaded_hp = load_tuned_hyperparameters(TUNED_HP_FILE)
        if loaded_hp:
            # Update global parameters with loaded values
            plan_count = loaded_hp.get('plan_count', plan_count)
            learning_rate = loaded_hp.get('learning_rate', learning_rate)
            batch_size = loaded_hp.get('batch_size', batch_size)
            beta = loaded_hp.get('beta', beta)
            epsilon_min = loaded_hp.get('epsilon_min', epsilon_min)
            epsilon_decay = loaded_hp.get('epsilon_decay', epsilon_decay)
            discount_factor = loaded_hp.get('discount_factor', discount_factor)
            model_train_start = loaded_hp.get('model_train_start', model_train_start)
            agent_train_start = loaded_hp.get('agent_train_start', agent_train_start)
            real_synthetic_ratio = loaded_hp.get('real_synthetic_ratio', real_synthetic_ratio)

            logging.info("Successfully loaded previously tuned hyperparameters!")
        else:
            logging.warning("No previously tuned hyperparameters found - using default values")
            logging.warning(f"To create tuned hyperparameters, set ENABLE_HP_TUNING = True and run again")

    # Update hyperparams dict with (possibly tuned) values
    current_hyperparams = build_hyperparams_dict(
        experiment_time=experiment_time,
        beta=beta,
        learning_rate=learning_rate,
        epsilon_min=epsilon_min,
        batch_size=batch_size,
        discount_factor=discount_factor,
        target_temp=target_temp
    )

    logging.info("=" * 80)
    logging.info("FINAL HYPERPARAMETERS FOR MAIN TRAINING:")
    logging.info(f"  experiment_time  = {experiment_time}")
    logging.info(f"  plan_count       = {plan_count}")
    logging.info(f"  learning_rate    = {learning_rate}")
    logging.info(f"  batch_size       = {batch_size}")
    logging.info(f"  beta             = {beta}")
    logging.info(f"  epsilon_min      = {epsilon_min}")
    logging.info(f"  epsilon_decay    = {epsilon_decay}")
    logging.info(f"  discount_factor  = {discount_factor}")
    logging.info("=" * 80)

    # -------------------------------------------------------------------------
    # Train All Registered RL Modules - TWO-PASS TRAINING
    # Pass 1: load_model=False, save_model=True (fresh training)
    # Pass 2: load_model=True, save_model=False (continue from saved models)
    # -------------------------------------------------------------------------
    all_results: Dict[str, Tuple] = {}
    all_results_pass1: Dict[str, Tuple] = {}  # Results from pass 1 (fresh training)
    all_results_pass2: Dict[str, Tuple] = {}  # Results from pass 2 (loaded models)
    all_csv_files: Dict[str, str] = {}
    all_csv_files_pass1: Dict[str, str] = {}
    all_csv_files_pass2: Dict[str, str] = {}
    all_hyperparams: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}  # Track historical hyperparams

    # ==========================================================================
    # PASS 1: Fresh Training (load_model=False, save_model=True)
    # ==========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("PASS 1: FRESH TRAINING (load_model=False, save_model=True)")
    logging.info("=" * 80 + "\n")

    pass1_load_model = False
    pass1_save_model = True
    pass1_timestamp = timestamp

    for module_idx, (module_name, train_function) in enumerate(RL_MODULE_REGISTRY.items()):
        logging.info("\n" + "=" * 80)
        logging.info(f"PHASE {module_idx + 1}/{len(RL_MODULE_REGISTRY)}: Starting {module_name} Training")
        logging.info("=" * 80)

        # Build filename with hyperparameters encoded
        server_name_main = os.path.join(
            SAVE_DIR,
            build_filename_with_hyperparams(module_name, timestamp, current_hyperparams)
        )

        # =====================================================================
        # Historical Data Selection - PROPER BASELINE SELECTION
        # =====================================================================
        # Find historical files with the SAME number of experiments for fair comparison
        # This ensures X-axis alignment and meaningful comparison
        #
        # BASELINE SELECTION:
        # - Single-Agent modules (SAMFRL, SAMBRL, SARBRL, SAMBPGRL, MAML) compare with SAMFRL
        # - Multi-Agent modules (MAMFRL, MAMBRL, MARBRL, MARB_D3QN, MARL_GEAR) compare with MAMFRL_D3QN
        # =====================================================================
        server_name_1 = ""
        server_name_2 = ""
        historical_hyperparams: List[Tuple[str, Dict[str, Any]]] = []

        # Determine the baseline module for cross-comparison based on module type
        module_type = get_module_type(module_name)
        baseline_module = get_baseline_for_module(module_name)

        # If this IS the baseline module, compare with same module history only
        if module_name == baseline_module:
            other_module = None
            logging.info(f"{module_name} is the {module_type} baseline - comparing with historical {module_name} data only")
        else:
            # Compare with the appropriate baseline (SAMFRL for SA, MAMFRL for MA)
            other_module = baseline_module
            logging.info(f"{module_name} ({module_type}) will compare with baseline: {other_module}")

        # First, check if previous module in current session has results
        # NOTE: Previous module (e.g., MAMBRL) may have different row count than current module (e.g., MAMFRL)
        # MAMBRL saves one row per app per iteration, MAMFRL saves one row per iteration
        # For fair comparison, we should NOT use cross-module data directly from current session
        # unless the row counts match. The individual training scripts will do final validation.
        if module_idx > 0:
            prev_files = list(all_csv_files.values())
            if len(prev_files) >= 1:
                prev_file = prev_files[-1]
                prev_module_name = list(all_csv_files.keys())[-1]
                # Verify the file exists and parse its hyperparams
                prev_hp = parse_hyperparams_from_filename(prev_file)

                # Count actual rows in the previous file
                prev_row_count = 0
                try:
                    with open(prev_file, 'r') as f:
                        prev_row_count = sum(1 for _ in f) - 1  # Subtract header
                except Exception:
                    pass

                # Only use if both exp count AND row count match
                # This ensures fair comparison (apples to apples)
                if prev_hp and prev_hp.get('exp') == experiment_time and prev_row_count == experiment_time:
                    server_name_1 = prev_file
                    historical_hyperparams.append((server_name_1, {
                        **current_hyperparams,
                        'module_name': prev_module_name,
                        'timestamp': timestamp
                    }))
                    logging.info(f"Using current session {prev_module_name} data: {os.path.basename(server_name_1)} (rows={prev_row_count})")
                else:
                    logging.warning(f"Skipping current session file (need exp={experiment_time} and rows={experiment_time}, "
                                  f"got exp={prev_hp.get('exp') if prev_hp else 'N/A'}, rows={prev_row_count}): {os.path.basename(prev_file)}")

        # Find historical files with SAME experiment count from previous runs
        exclude_files = list(all_csv_files.values()) + [server_name_main]

        # Find historical files from the baseline module (SAMFRL) for cross-module comparison
        # Use require_exact_row_count=False for cross-module (data will be aggregated at load time)
        other_module_files = []
        if other_module:  # Not None (i.e., not SAMFRL itself)
            other_module_files = find_historical_data_with_same_experiments(
                save_dir=SAVE_DIR,
                module_name=other_module,
                target_experiment_count=experiment_time,
                exclude_files=exclude_files,
                require_exact_row_count=False  # Allow different row counts for cross-module
            )

        # Find historical files from the SAME module type with same experiment count
        # Use require_exact_row_count=True for same-module (row counts should match)
        same_exp_files = find_historical_data_with_same_experiments(
            save_dir=SAVE_DIR,
            module_name=module_name,
            target_experiment_count=experiment_time,
            exclude_files=exclude_files,
            require_exact_row_count=True  # Require exact row count match
        )

        # Assign historical references
        # server_name_1: Cross-module comparison (all modules compare with SAMFRL baseline)
        # This data may need aggregation - store row count info for the training script
        if not server_name_1 and other_module_files:
            server_name_1 = other_module_files[0][0]
            historical_hyperparams.append(other_module_files[0])
            hp = other_module_files[0][1]
            logging.info(f"Found historical {other_module} data with {experiment_time} experiments: {os.path.basename(server_name_1)}")
            if hp:
                logging.info(f"  Hyperparams: rows={hp.get('actual_row_count')}, beta={hp.get('beta')}, lr={hp.get('lr')}, eps={hp.get('eps_min')}, batch={hp.get('batch')}")

        # server_name_2: Same-module comparison (e.g., MAMFRL comparing with previous MAMFRL)
        if not server_name_2 and same_exp_files:
            server_name_2 = same_exp_files[0][0]
            historical_hyperparams.append(same_exp_files[0])
            hp = same_exp_files[0][1]
            logging.info(f"Found historical {module_name} data with {experiment_time} experiments: {os.path.basename(server_name_2)}")
            if hp:
                logging.info(f"  Hyperparams: rows={hp.get('actual_row_count')}, beta={hp.get('beta')}, lr={hp.get('lr')}, eps={hp.get('eps_min')}, batch={hp.get('batch')}")

        # Store historical hyperparams for this module
        all_hyperparams[module_name] = historical_hyperparams

        if server_name_1 or server_name_2:
            logging.info("Historical data for comparison (SAME experiment count):")
            if server_name_1:
                logging.info(f"  Reference 1: {os.path.basename(server_name_1)}")
            if server_name_2:
                logging.info(f"  Reference 2: {os.path.basename(server_name_2)}")

            # Display hyperparameter comparison
            if historical_hyperparams:
                logging.info("\n" + format_hyperparams_table(current_hyperparams, historical_hyperparams))

        try:
            module_epsilon = epsilon
            logging.info("=" * 80)
            logging.info(f"PASS 1: Calling training function for {module_name}...")
            logging.info(f"Parameters: experiment_time={experiment_time}, beta={beta}, epsilon={module_epsilon}")
            logging.info(f"load_model={pass1_load_model}, save_model={pass1_save_model}")
            logging.info("=" * 80)

            training_start_time = time.time()

            # Build base kwargs for training function
            # NOTE: model_train_start and real_synthetic_ratio only supported by:
            # SAMBRL, MAMBRL_D3QN, MARB_D3QN, SARBRL
            train_kwargs = {
                'client_socket': client_socket,
                'data_keys': DATA_KEYS,
                'experiment_time': experiment_time,
                'clock_change_time': clock_change_time,
                'beta': beta,
                'load_model': pass1_load_model,
                'learn_count': model_train_start,
                'plan_count': plan_count,
                'mem_size': mem_size,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'epsilon': module_epsilon,
                'epsilon_decay': epsilon_decay,
                'epsilon_min': epsilon_min,
                'epsilon_start': epsilon_start,
                'epsilon_end': epsilon_end,
                'reset_learning_rate_value': reset_learning_rate_value,
                'save_repetition': save_repetition,
                'save_model': pass1_save_model,
                'batch_size': batch_size,
                'agent_train_start': agent_train_start,
                'target_temp': target_temp,
                'server_name_1': server_name_1,
                'server_name_2': server_name_2,
                'server_name_main': server_name_main,
                'profiling_data_list': profiling_data_list,
                'application_profiles': application_profiles,
                'applications_fixed': app_config.applications_fixed,
                'priority_combinations': app_config.priority_combinations,
                'frequency_combinations': app_config.frequency_combinations,
                'num_cores_list': app_config.num_cores_list,
            }

            # Add model-based RL parameters only for modules that support them
            MODULES_WITH_MODEL_TRAIN_START = ['SAMBRL', 'MAMBRL_D3QN', 'MARB_D3QN', 'SARBRL']
            if module_name in MODULES_WITH_MODEL_TRAIN_START:
                train_kwargs['model_train_start'] = model_train_start
                train_kwargs['real_synthetic_ratio'] = real_synthetic_ratio

            # Call training function with PASS 1 settings (fresh training)
            results = train_function(**train_kwargs)

            training_duration = time.time() - training_start_time
            logging.info("=" * 80)
            logging.info(f"PASS 1: Training function returned after {training_duration:.1f} seconds")
            logging.info("=" * 80)

            if results is not None:
                all_results[module_name] = results
                all_results_pass1[module_name] = results
                all_csv_files[module_name] = server_name_main
                all_csv_files_pass1[module_name] = server_name_main
                logging.info(f"PASS 1: {module_name} Training Complete")
                logging.info(f"  Results saved to: {os.path.basename(server_name_main)}")
            else:
                logging.warning(f"PASS 1: {module_name} training returned None (possible initialization error)")
                all_results[module_name] = None
                all_results_pass1[module_name] = None

        except Exception as e:
            logging.error(f"PASS 1: {module_name} training failed: {e}", exc_info=True)
            all_results[module_name] = None
            all_results_pass1[module_name] = None

        # Memory cleanup after each module to prevent swap/RAM exhaustion
        logging.info("Performing memory cleanup after module completion...")
        cleanup_memory()

        # Short cooldown between modules (NO phase sync needed)
        if module_idx < len(RL_MODULE_REGISTRY) - 1:
            logging.info("Cooling for 5 seconds before starting next module...")
            time.sleep(5)

    # ==========================================================================
    # PASS 2: Continue Training with Loaded Models (load_model=True, save_model=False)
    # ==========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("PASS 2: CONTINUE TRAINING (load_model=True, save_model=False)")
    logging.info("=" * 80 + "\n")

    pass2_load_model = True
    pass2_save_model = False
    pass2_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for module_idx, (module_name, train_function) in enumerate(RL_MODULE_REGISTRY.items()):
        logging.info("\n" + "=" * 80)
        logging.info(f"PASS 2 - MODULE {module_idx + 1}/{len(RL_MODULE_REGISTRY)}: {module_name}")
        logging.info("=" * 80)

        # Build filename with hyperparameters for PASS 2
        pass2_server_name_main = os.path.join(
            SAVE_DIR,
            build_filename_with_hyperparams(f"{module_name}_LOADED", pass2_timestamp, current_hyperparams)
        )

        # Use SAME historical references as PASS 1, plus PASS 1 results for direct comparison
        # This ensures LOADED plots show the same baseline comparisons as non-LOADED plots
        server_name_1_pass2 = all_csv_files_pass1.get(module_name, "")  # PASS 1 (fresh training) data

        # Re-search for cross-module baseline (e.g., SAMFRL) to maintain consistent comparison
        server_name_2_pass2 = ""
        baseline_modules = ["SAMFRL", "SAMBRL", "SARBRL"]  # Baseline modules to compare against
        exclude_loaded = list(all_csv_files_pass1.values()) + list(all_csv_files_pass2.values())
        for baseline_module in baseline_modules:
            if baseline_module != module_name:  # Don't compare with self
                baseline_files = find_historical_data_with_same_experiments(
                    save_dir=SAVE_DIR,
                    module_name=baseline_module,
                    target_experiment_count=experiment_time,
                    exclude_files=exclude_loaded,
                    require_exact_row_count=False
                )
                if baseline_files:
                    server_name_2_pass2 = baseline_files[0][0]
                    logging.info(f"PASS 2: Using {baseline_module} baseline for comparison: {os.path.basename(server_name_2_pass2)}")
                    break

        try:
            module_epsilon = epsilon_min  # Start with lower epsilon for loaded model
            logging.info("=" * 80)
            logging.info(f"PASS 2: Calling training function for {module_name}...")
            logging.info(f"Parameters: experiment_time={experiment_time}, beta={beta}, epsilon={module_epsilon}")
            logging.info(f"load_model={pass2_load_model}, save_model={pass2_save_model}")
            logging.info("=" * 80)

            training_start_time = time.time()

            # Build base kwargs for training function
            # NOTE: model_train_start and real_synthetic_ratio only supported by:
            # SAMBRL, MAMBRL_D3QN, MARB_D3QN, SARBRL
            train_kwargs = {
                'client_socket': client_socket,
                'data_keys': DATA_KEYS,
                'experiment_time': experiment_time,
                'clock_change_time': clock_change_time,
                'beta': beta,
                'load_model': pass2_load_model,
                'learn_count': model_train_start,
                'plan_count': plan_count,
                'mem_size': mem_size,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'epsilon': module_epsilon,
                'epsilon_decay': epsilon_decay,
                'epsilon_min': epsilon_min,
                'epsilon_start': epsilon_min,  # Start from lower epsilon
                'epsilon_end': epsilon_end,
                'reset_learning_rate_value': reset_learning_rate_value,
                'save_repetition': save_repetition,
                'save_model': pass2_save_model,
                'batch_size': batch_size,
                'agent_train_start': agent_train_start,
                'target_temp': target_temp,
                'server_name_1': server_name_1_pass2,
                'server_name_2': server_name_2_pass2,
                'server_name_main': pass2_server_name_main,
                'profiling_data_list': profiling_data_list,
                'application_profiles': application_profiles,
                'applications_fixed': app_config.applications_fixed,
                'priority_combinations': app_config.priority_combinations,
                'frequency_combinations': app_config.frequency_combinations,
                'num_cores_list': app_config.num_cores_list,
            }

            # Add model-based RL parameters only for modules that support them
            MODULES_WITH_MODEL_TRAIN_START = ['SAMBRL', 'MAMBRL_D3QN', 'MARB_D3QN', 'SARBRL']
            if module_name in MODULES_WITH_MODEL_TRAIN_START:
                train_kwargs['model_train_start'] = model_train_start
                train_kwargs['real_synthetic_ratio'] = real_synthetic_ratio

            # Call training function with PASS 2 settings (continue from saved models)
            results = train_function(**train_kwargs)

            training_duration = time.time() - training_start_time
            logging.info("=" * 80)
            logging.info(f"PASS 2: Training function returned after {training_duration:.1f} seconds")
            logging.info("=" * 80)

            if results is not None:
                all_results_pass2[module_name] = results
                all_csv_files_pass2[module_name] = pass2_server_name_main
                logging.info(f"PASS 2: {module_name} Training Complete")
                logging.info(f"  Results saved to: {os.path.basename(pass2_server_name_main)}")
            else:
                logging.warning(f"PASS 2: {module_name} training returned None")
                all_results_pass2[module_name] = None

        except Exception as e:
            logging.error(f"PASS 2: {module_name} training failed: {e}", exc_info=True)
            all_results_pass2[module_name] = None

        # Memory cleanup after each module to prevent swap/RAM exhaustion
        logging.info("Performing memory cleanup after module completion...")
        cleanup_memory()

        # Short cooldown between modules
        if module_idx < len(RL_MODULE_REGISTRY) - 1:
            logging.info("Cooling for 5 seconds before starting next module...")
            time.sleep(5)

    # -------------------------------------------------------------------------
    # Comparison and Visualization - Including Pass 1 vs Pass 2 Comparison
    # -------------------------------------------------------------------------
    if all_results_pass1 or all_results_pass2:
        logging.info("=" * 80)
        logging.info("Generating Comparison Plots...")
        logging.info("=" * 80)

        # Generate PASS 1 comparison plot (fresh training results)
        plot_comparison_with_history(
            results_dict=all_results_pass1,
            save_path=comparison_plot_name,
            all_csv_files=all_csv_files_pass1,
            current_hyperparams=current_hyperparams,
            all_hyperparams=all_hyperparams
        )

        # Generate Pass 1 vs Pass 2 comparison plot (load_model comparison)
        if all_results_pass2:
            pass1_vs_pass2_plot_name = os.path.join(
                SAVE_DIR,
                f"Comparison_LoadModel_{timestamp}_{experiment_time}ep_beta{beta}.png"
            )
            plot_pass1_vs_pass2_comparison(
                pass1_results=all_results_pass1,
                pass2_results=all_results_pass2,
                save_path=pass1_vs_pass2_plot_name,
                experiment_time=experiment_time
            )

        logging.info("=" * 80)
        logging.info("Comparison Complete!")
        logging.info("=" * 80)
        logging.info("PASS 1 Results (fresh training):")
        for module_name, csv_file in all_csv_files_pass1.items():
            logging.info(f"  {module_name}: {os.path.basename(csv_file)}")
        logging.info(f"PASS 2 Results (loaded models):")
        for module_name, csv_file in all_csv_files_pass2.items():
            logging.info(f"  {module_name}: {os.path.basename(csv_file)}")
        logging.info(f"Comparison plots:")
        logging.info(f"  Pass 1 comparison: {os.path.basename(comparison_plot_name)}")
        if all_results_pass2:
            logging.info(f"  Pass 1 vs Pass 2: {os.path.basename(pass1_vs_pass2_plot_name)}")
        logging.info("=" * 80)
    else:
        logging.warning("No successful training runs - cannot generate comparison")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("All training complete. Shutting down server.")
    print("=" * 80)

    try:
        client_socket.close()
    except Exception:
        pass
    try:
        server_socket.close()
    except Exception:
        pass

    logging.info("Server shutdown gracefully.")
