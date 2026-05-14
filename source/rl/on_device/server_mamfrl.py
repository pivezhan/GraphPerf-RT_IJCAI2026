#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAMFRL with D3QN - Model-Free Multi-Agent RL for Jetson TX2
FIXED VERSION - Compatible with client_evaluate_tx2.py

KEY FIXES FROM mamfrl_d3qn_tx2.py:
=================================
1. Communication now matches server_mamfrl_d3qn.py + client_evaluate_tx2.py pattern
2. Sends proper application messages: {'applications': [...], 'run_mode': 'parallel'}
3. app_instance format includes all required fields (benchmark, variant, input_arg, app_args)
4. Training loop restructured: SEND apps first, then RECEIVE results

CLIENT MESSAGE FORMAT (client_evaluate_tx2.py expects):
======================================================
{
    "id": int,                 # Application identifier
    "benchmark": str,          # e.g., "fft", "gemm"
    "variant": str,            # e.g., "bin-omp-tasks" for BOTS, "" for PolyBench
    "input_arg": str,          # e.g., "5" for BOTS, "STANDARD" for PolyBench
    "app_args": str,           # CRITICAL: same as input_arg (client uses this!)
    "frequencies": list,       # List of frequency indices per core
    "num_cores": int,          # Number of cores allocated
    "cores": str,              # Comma-separated core list (e.g., "1,2,3")
    "priority": int,           # RT priority value
    "action": str              # "profile" or "run"
}

TRAINING FEATURES (preserved from original):
============================================
1. Adaptive reward system with improvement bonuses
2. Best performance tracking and curriculum learning
3. Higher epsilon minimum (0.10) for continued exploration
4. Exploration boost when stuck
5. Progress bonuses and dynamic target adaptation
6. MODEL-FREE: No environment model, no planning, only real experiences

DIFFERENCES FROM MAMBRL:
========================
- NO EnvironmentModel class
- NO planning/synthetic data generation
- Only trains on real replay buffer data
- Simpler structure without generative memories
"""

import logging
import json
import time
import numpy as np
import random
import csv
import os
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import re
import socket
import threading
from copy import deepcopy
from scipy import stats

# Import unified plotting module
from live_plotter import (
    OnlinePlotter, create_data_map,
    load_historical_data, extract_server_name, parse_hyperparams_from_filename
)
from server_combined import DATA_KEYS

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of this script for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "save_model")

STATE_SIZE = None


# =============================================================================
# Historical Data Loading and Plotting Utilities
# =============================================================================
def load_historical_data(filepath, data_keys, target_rows=None):
    """Load historical training data from CSV for comparison plotting.

    Args:
        filepath: Path to the CSV file
        data_keys: List of column names to load
        target_rows: If specified and data has more rows, aggregate to this count
                    (for cross-module comparison where MAMBRL has 3x rows)

    Returns:
        Dictionary mapping data_keys to lists of values
    """
    data = {key: [] for key in data_keys}
    if os.path.exists(filepath):
        logging.info(f"Loading historical data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            actual_rows = len(df)

            # Check if we need to aggregate (cross-module comparison)
            if target_rows and actual_rows > target_rows and actual_rows % target_rows == 0:
                # Aggregate by taking mean of each group
                # e.g., 300 rows -> 100 rows (MAMBRL has 3 rows per iteration for 3 apps)
                group_size = actual_rows // target_rows
                logging.info(f"Aggregating {actual_rows} rows to {target_rows} (group_size={group_size})")

                for key in data_keys:
                    if key in df.columns:
                        values = df[key].tolist()
                        # Average every group_size values
                        aggregated = []
                        for i in range(0, len(values), group_size):
                            group = values[i:i+group_size]
                            aggregated.append(np.mean(group))
                        data[key] = aggregated
            else:
                # No aggregation needed - load directly
                for key in data_keys:
                    if key in df.columns:
                        data[key] = df[key].tolist()

        except Exception as e:
            logging.warning(f"Failed to load historical data: {e}")
    else:
        logging.warning(f"No historical data found at {filepath}")
    return data


def extract_server_name(tuning_name):
    """Extract server algorithm name from filename for plot legend.

    Returns descriptive names like 'MAMBRL (Model-Based)' or 'MAMFRL (Model-Free)'
    """
    if not tuning_name:
        return "Unknown"

    basename = os.path.basename(tuning_name)

    # Check for MAMBRL or MAMFRL in filename
    if "MAMBRL" in basename.upper():
        return "MAMBRL (Model-Based)"
    elif "MAMFRL" in basename.upper():
        return "MAMFRL (Model-Free)"

    # Fallback: try old pattern
    match = re.search(r"server_(.*?)_\d+", tuning_name)
    if match:
        return match.group(1)

    # Last resort: use filename prefix
    if '_' in basename:
        return basename.split('_')[0]

    return "Historical"


def parse_hyperparams_from_filename(filename):
    """
    Parse hyperparameters from a filename.
    Returns dict with keys: exp, beta, lr, eps_min, batch, module_name
    Returns None if parsing fails.

    Supports filenames like:
    - MAMBRL_D3QN_20251204_155454_100ep_beta1_lr0.05_eps0.1_batch32.csv (new format)
    - MAMBRL_D3QN_20251204_155454_100ep_beta1.csv (old format)
    """
    basename = os.path.basename(filename)

    # Try new format: MODULE_YYYYMMDD_HHMMSS_EXPep_betaB_lrL_epsE_batchB.csv
    # Module name can contain underscores (e.g., MAMBRL_D3QN)
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
    # Module name can contain underscores (e.g., MAMBRL_D3QN)
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


def find_historical_data_with_same_experiments(save_dir, module_name, target_experiment_count, exclude_files=None):
    """
    Find historical CSV files for the given module with the same experiment count.
    Returns list of (filepath, hyperparams_dict) tuples, sorted by timestamp (newest first).
    """
    import csv as csv_module
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

            hyperparams = parse_hyperparams_from_filename(filename)
            if hyperparams is None:
                continue

            if module_name in hyperparams.get('module_name', ''):
                if hyperparams.get('exp') == target_experiment_count:
                    filepath = os.path.join(save_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            reader = csv_module.reader(f)
                            row_count = sum(1 for _ in reader) - 1
                        if row_count == target_experiment_count:
                            matching_files.append((filepath, hyperparams))
                    except Exception:
                        matching_files.append((filepath, hyperparams))

        matching_files.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
        logging.info(f"Found {len(matching_files)} historical files for {module_name} with {target_experiment_count} experiments")
        return matching_files

    except Exception as e:
        logging.error(f"Error finding historical data: {e}")
        return []


def priority_tuple_to_int(tup):
    """Convert priority tuple to integer for logging."""
    return int(''.join(str(x) for x in tup))

# =============================================================================
# NEW: Best Performance Tracker for Adaptive Rewards
# =============================================================================
class BestPerformanceTracker:
    """Tracks best performance for adaptive rewards and curriculum learning."""
    
    def __init__(self):
        self.best_makespan = float('inf')
        self.best_energy = float('inf')
        self.best_temperature = float('inf')
        self.iterations_since_improvement = 0
        self.improvement_history = []
        
    def update(self, makespan, energy, temperature):
        """Update best performance and return if improved."""
        improved = False
        
        if makespan < self.best_makespan:
            improvement_pct = (self.best_makespan - makespan) / self.best_makespan * 100
            logging.info(f"✅ NEW BEST MAKESPAN: {makespan:.3f}s ({improvement_pct:.1f}% improvement)")
            self.best_makespan = makespan
            self.iterations_since_improvement = 0
            self.improvement_history.append(improvement_pct)
            improved = True
        
        if energy < self.best_energy:
            self.best_energy = energy
            improved = True
        
        if temperature < self.best_temperature:
            self.best_temperature = temperature
        
        if not improved:
            self.iterations_since_improvement += 1
        
        return improved
    
    def needs_exploration_boost(self, threshold=20):
        """Check if stuck and needs exploration boost."""
        return self.iterations_since_improvement > threshold
    
    def get_stats(self):
        """Get performance statistics."""
        return {
            'best_makespan': self.best_makespan,
            'best_energy': self.best_energy,
            'iterations_since_improvement': self.iterations_since_improvement,
            'total_improvements': len(self.improvement_history)
        }


# =============================================================================
# Utility Functions
# =============================================================================
def subset_to_int(subset_str: str) -> int:
    """Convert comma-separated core string to integer bitmask representation."""
    subset_int = 0
    cores = subset_str.split(',')
    for core in cores:
        core = core.strip()
        if core.isdigit():
            core_id = int(core)
            subset_int |= (1 << core_id)
    return subset_int


def get_temperature_with_fallbacks(d, target_temp=50):
    """Get temperature from profiling data with multiple fallbacks."""
    # Try avg_temp_after first
    temp = d.get('avg_temp_after')
    if temp is not None and temp > 0:
        return float(temp)

    # Try thermal zones
    for zone in range(10):
        zone_temp = d.get(f'thermal_zone{zone}')
        if zone_temp is not None and zone_temp > 0:
            return float(zone_temp)

    # Try specific CPU temps
    cpu_temp = d.get('CPU57_temp_after')
    if cpu_temp is not None and cpu_temp > 0:
        return float(cpu_temp)

    mcpu_temp = d.get('MCPU_temp')
    if mcpu_temp is not None and mcpu_temp > 0:
        return float(mcpu_temp)

    # Default to target_temp
    return float(target_temp)


def normalize_profiling_data(d: dict) -> dict:
    """
    Normalize profiling data from client to ensure all expected keys exist.
    Maps new format (thermal_zone*, energy_*_j) to old format (CPU57_temp_after, etc.)
    """
    if d is None:
        return {}

    # Create a copy to avoid modifying original
    data = dict(d)

    # --- Compute avg_temp_after from thermal zones if not present ---
    if data.get('avg_temp_after') is None or data.get('avg_temp_after', 0) <= 0:
        temps = []
        for zone in range(10):
            zone_temp = data.get(f'thermal_zone{zone}')
            if zone_temp is not None and zone_temp > 0:
                temps.append(float(zone_temp))
        if temps:
            data['avg_temp_after'] = sum(temps) / len(temps)
        else:
            for key in ['CPU57_temp_after', 'MCPU_temp', 'CPU_temp']:
                if data.get(key) is not None and data.get(key, 0) > 0:
                    data['avg_temp_after'] = float(data.get(key))
                    break
            else:
                data['avg_temp_after'] = 50.0

    # --- Map energy keys: new format -> old format ---
    energy_mapping = {
        'CPU57_energy_joules': ['energy_cpu_j', 'energy_CPU_j'],
        'Denver_energy_joules': ['energy_denver_j', 'energy_Denver_j'],
        'System_energy_joules': ['energy_system_j', 'energy_System_j', 'energy_main_j'],
        'GPU_energy_joules': ['energy_gpu_j', 'energy_GPU_j'],
        'DDR_energy_joules': ['energy_ddr_j', 'energy_DDR_j'],
    }
    for old_key, new_keys in energy_mapping.items():
        if data.get(old_key) is None or data.get(old_key, 0) <= 0:
            for new_key in new_keys:
                if data.get(new_key) is not None and data.get(new_key, 0) > 0:
                    data[old_key] = float(data.get(new_key))
                    break
            else:
                data[old_key] = 0.0

    # --- Map temperature keys: thermal zones -> old format ---
    temp_mapping = {
        'CPU57_temp_after': ['thermal_zone0', 'thermal_zone1'],
        'Denver_temp_after': ['thermal_zone2', 'thermal_zone3'],
        'System_temp_after': ['thermal_zone4', 'thermal_zone5'],
        'GPU_temp_after': ['thermal_zone1', 'thermal_zone6'],
        'DDR_temp_after': ['thermal_zone7', 'thermal_zone3'],
    }
    for old_key, zone_keys in temp_mapping.items():
        if data.get(old_key) is None or data.get(old_key, 0) <= 0:
            for zone_key in zone_keys:
                if data.get(zone_key) is not None and data.get(zone_key, 0) > 0:
                    data[old_key] = float(data.get(zone_key))
                    break
            else:
                data[old_key] = data.get('avg_temp_after', 50.0)

    # --- Ensure temp deltas exist (default to 0) ---
    for key in ['CPU57_temp_delta', 'Denver_temp_delta', 'System_temp_delta', 'GPU_temp_delta', 'DDR_temp_delta', 'avg_temp_delta']:
        if data.get(key) is None:
            data[key] = 0.0

    # --- Ensure perf counters exist with defaults ---
    perf_keys = ['cycles', 'cache_references', 'cache_misses', 'branch_instructions',
                 'task_clock', 'context_switches', 'minor_faults', 'major_faults',
                 'branch_misses', 'branches', 'instructions', 'page_faults', 'cpu_clock']
    for key in perf_keys:
        if data.get(key) is None:
            data[key] = 0

    # --- Ensure other expected keys exist ---
    if data.get('utilization') is None:
        data['utilization'] = 0.0
    if data.get('parallelism_level') is None:
        data['parallelism_level'] = 1.0
    if data.get('time_elapsed') is None:
        data['time_elapsed'] = 0.0
    if data.get('total_energy_consumption') is None:
        data['total_energy_consumption'] = 0.0

    return data


def parse_state(profiling_data):
    """
    Parse profiling data dict into a state vector, with thermal and energy data.

    Returns:
        state: np.array, the state vector for the profiler agent
        total_energy_consumption: float
        avg_temp_after: float
        time_elapsed: float
        branch_misses: int
        cache_misses: int
        target_makespan: float  (per-app min makespan, if present)
        target_energy: float    (per-app energy baseline, if present)
        parallelism_level: float
    """
    global STATE_SIZE
    try:
        # Normalize data to ensure all expected keys exist
        profiling_data = normalize_profiling_data(profiling_data)
        # Frequency level (0-11 for TX2)
        # FIXED: Handle case where frequencies exists but is empty []
        # .get('frequencies', [0]) returns [] if key exists with empty list
        frequencies = profiling_data.get('frequencies') or [0]
        c_c = frequencies[0] if frequencies else 0

        # Core allocation
        subset_str = profiling_data.get('cores', '')
        subset_int = subset_to_int(subset_str)

        # Performance metrics
        utilization = profiling_data.get('utilization', 0.0)
        time_elapsed = profiling_data.get('time_elapsed', 0.0)
        total_energy_consumption = profiling_data.get('total_energy_consumption', 0.0)

        # Thermal metrics
        avg_temp_after = profiling_data.get('avg_temp_after', 0.0)
        if avg_temp_after is None:
            avg_temp_after = 0.0
            logging.warning("avg_temp_after is None, using 0.0")

        avg_temp_delta = profiling_data.get('avg_temp_delta', 0.0)
        if avg_temp_delta is None:
            avg_temp_delta = 0.0
            logging.warning("avg_temp_delta is None, using 0.0")

        # Target metrics (per-application)
        target_makespan = profiling_data.get('makespan_all_cores_frequency_11', time_elapsed)
        target_energy = profiling_data.get('energy_all_cores_frequency_0', total_energy_consumption)
        parallelism_level = profiling_data.get('parallelism_level', 1.0)

        # Validate per-app target values
        if target_makespan is None or target_makespan <= 0:
            target_makespan = time_elapsed
            logging.warning(f"Invalid per-app target_makespan, using current makespan: {target_makespan:.3f}s")

        if target_energy is None or target_energy <= 0:
            target_energy = total_energy_consumption
            logging.warning(f"Invalid per-app target_energy, using current energy: {target_energy:.3f}J")

        # Performance counters
        cycles = profiling_data.get('cycles', 0)
        cache_references = profiling_data.get('cache_references', 0)
        cache_misses = profiling_data.get('cache_misses', 0)
        branch_instructions = profiling_data.get('branch_instructions', 0)
        task_clock = profiling_data.get('task_clock', 0.0)
        context_switches = profiling_data.get('context_switches', 0)
        minor_faults = profiling_data.get('minor_faults', 0)
        major_faults = profiling_data.get('major_faults', 0)
        branch_misses = profiling_data.get('branch_misses', 0)
        branches = profiling_data.get('branches', 0)
        instructions = profiling_data.get('instructions', 0)
        page_faults = profiling_data.get('page_faults', 0)
        cpu_clock = profiling_data.get('cpu_clock', 0.0)

        # Per-source energy consumption
        energy_consumption = []
        energy_keys = ['energy_system_j', 'energy_cpu_j', 'energy_denver_j', 'energy_gpu_j', 'energy_ddr_j']
        for key in energy_keys:
            energy_consumption.append(profiling_data.get(key, 0.0))

        # Per-zone temperatures
        temps_after = []
        for i in range(10):
            temp_key = f'thermal_zone{i}'
            if temp_key in profiling_data:
                temp_val = profiling_data.get(temp_key, 0.0)
                temps_after.append(float(temp_val) if temp_val is not None else 0.0)
            else:
                temps_after.append(0.0)

        # Temperature deltas
        temp_deltas = []
        for i in range(min(5, len(temps_after))):
            temp_key = f'temp_delta{i}'
            temp_deltas.append(profiling_data.get(temp_key, 0.0))

        # Assemble state vector
        state = [
            float(c_c),
            float(subset_int),
            float(utilization),
            float(cycles),
            float(cache_references),
            float(cache_misses),
            float(branch_instructions),
            float(task_clock),
            float(context_switches),
            float(minor_faults),
            float(major_faults),
            float(branch_misses),
            float(branches),
            float(instructions),
            float(page_faults),
            float(cpu_clock),
            float(time_elapsed),
        ]
        state.extend([float(e) for e in energy_consumption])
        state.extend([float(t) for t in temps_after])
        state.extend([float(td) for td in temp_deltas])
        state.extend([
            float(target_makespan),
            float(target_energy),
            float(avg_temp_delta),
            float(parallelism_level)
        ])

        state = np.array(state, dtype=np.float32)
        if STATE_SIZE is None:
            STATE_SIZE = len(state)

        return (
            state,
            total_energy_consumption,
            avg_temp_after,
            time_elapsed,
            branch_misses,
            cache_misses,
            target_makespan,
            target_energy,
            parallelism_level
        )

    except Exception as e:
        logging.error(f"Error parsing state: {e}", exc_info=True)
        return None


# =============================================================================
# Reward Functions (SAME AS MAMBRL D3QN)
# =============================================================================
def get_reward_profiler(makespan, avg_energy_consumption, target_makespan, target_energy, 
                        beta=1.0, best_makespan=None, iteration=0):
    """
    REVISED: Profiler reward with adaptive learning and improvement bonuses.
    
    Key improvements:
    - Uses best_makespan as adaptive baseline (curriculum learning)
    - Rewards improvements over previous best
    - Provides meaningful learning signal even when far from target
    - Adds progress bonuses for getting closer to target
    - Scales to reasonable range to prevent value explosion
    
    Returns:
        reward: float, the reward value
        improved: bool, whether this is an improvement over best
    """
    eps = 1e-6

    if target_makespan is None or target_makespan <= 0:
        target_makespan = max(makespan, eps)
        logging.warning(f"Invalid target_makespan in profiler reward, using current makespan: {target_makespan:.3f}s")

    if target_energy is None or target_energy <= 0:
        target_energy = max(avg_energy_consumption, eps)
        logging.warning(f"Invalid target_energy in profiler reward, using current energy: {target_energy:.3f}J")
    
    if best_makespan is None:
        best_makespan = makespan

    # Normalized ratios
    norm_makespan = float(makespan) / (target_makespan + eps)
    norm_energy = float(avg_energy_consumption) / (target_energy + eps)

    # Keep them non-negative and finite
    norm_makespan = max(norm_makespan, 0.0)
    norm_energy = max(norm_energy, 0.0)

    # Log-shaping to avoid huge unbounded costs
    ms_term = np.log1p(norm_makespan)
    en_term = np.log1p(norm_energy)
    
    # Cap at reasonable values to prevent explosion
    ms_term = min(ms_term, 3.0)
    en_term = min(en_term, 3.0)

    # Dynamic weighting: far from target => ignore energy
    if norm_makespan > 2.0:
        # Very far from target
        w_m = 1.0
        w_e = 0.0
    elif norm_makespan > 1.5:
        # Moderately far from target
        w_m = 0.8
        w_e = 0.2
    else:
        # Near target: use beta
        beta = max(beta, 0.0)
        w_m = beta / (beta + 1.0)
        w_e = 1.0 / (beta + 1.0)

    # Base performance cost
    base_cost = w_m * ms_term + w_e * en_term
    base_cost = min(base_cost, 5.0)  # Soft clamp

    # CRITICAL FIX: Add improvement bonus
    improvement_bonus = 0.0
    improved = False
    if best_makespan < float('inf') and makespan < best_makespan:
        improvement_ratio = (best_makespan - makespan) / best_makespan
        improvement_bonus = min(0.5, improvement_ratio * 2.0)  # Up to +0.5 bonus
        improved = True
        logging.info(f"🎯 IMPROVEMENT: {improvement_ratio*100:.1f}% better than best, bonus={improvement_bonus:.3f}")
    
    # CRITICAL FIX: Add progress bonus for getting closer to target
    progress_bonus = 0.0
    if norm_makespan < 1.5:  # Only when reasonably close
        progress_factor = max(0, 1.5 - norm_makespan) / 1.5
        progress_bonus = progress_factor * 0.3  # Up to +0.3
    
    # Final reward: negative cost + bonuses
    reward = -base_cost + improvement_bonus + progress_bonus
    
    # Clip to reasonable range
    reward = np.clip(reward, -3.0, 1.0)

    logging.debug(
        "Profiler reward (ADAPTIVE):\n"
        f"  makespan={makespan:.3f}, best={best_makespan:.3f}, target={target_makespan:.3f}\n"
        f"  energy={avg_energy_consumption:.3f}, target={target_energy:.3f}\n"
        f"  norm_ms={norm_makespan:.3f}, norm_en={norm_energy:.3f}\n"
        f"  base_cost={base_cost:.3f}, improvement_bonus={improvement_bonus:.3f}, progress_bonus={progress_bonus:.3f}\n"
        f"  REWARD={reward:.3f}, improved={improved}"
    )

    return float(reward), improved


def get_reward_thermal(avg_temperature, avg_temperature_prev, target_temp,
                       overheat_slope=0.1, cross_penalty=0.0):
    """
    Temperature agent reward (constraint-based, linear):

    - If avg_temperature <= target_temp:
        r_temp = 0       (no penalty, no bonus)
    - If avg_temperature >  target_temp:
        r_temp = -k * (avg_temperature - target_temp)

    where k = overheat_slope.

    Optionally, an extra penalty can be added when crossing from below
    to above the threshold (cross_penalty), but default is 0.0.
    """
    if avg_temperature <= target_temp:
        reward = 0.0
    else:
        overshoot = avg_temperature - target_temp
        reward = -min(5.0, overheat_slope * overshoot)

    if avg_temperature_prev < target_temp and avg_temperature > target_temp:
        reward -= cross_penalty

    logging.debug(
        "Thermal reward:\n"
        f"  current={avg_temperature:.2f}°C, prev={avg_temperature_prev:.2f}°C, "
        f"target={target_temp:.2f}°C, reward={reward:.3f}"
    )

    return float(reward)


def get_reward_priority(makespan, target_makespan):
    """
    Legacy priority reward (currently unused in the reward system):

        R3 = (target_makespan - makespan) / target_makespan
    """
    if target_makespan is None or target_makespan <= 0:
        logging.warning(f"Invalid target_makespan in priority reward: {target_makespan}")
        return 0.0

    diff = target_makespan - makespan
    reward = diff / float(target_makespan)
    return float(reward)


# =============================================================================
# Replay Buffer
# =============================================================================
class ReplayBuffer():
    def __init__(self, max_size, input_shape, action_size):
        self.mem_cntr = 0
        self.mem_size = max_size
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.action_size = action_size
        self.initialized = False

    def store_observations(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1
        if self.mem_cntr > 1:
            self.initialized = True

    def sample_observation(self, batch_size, random=True):
        max_mem = min(self.mem_cntr, self.mem_size)
        if max_mem < batch_size:
            return None
        if random:
            batch = np.random.choice(max_mem, batch_size, replace=False)
        else:
            batch_size = min(batch_size, max_mem)
            batch = np.arange(batch_size)
        return list(zip(
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch]
        ))

    def max_normalize(self, input_array, mode='state'):
        input_array = np.array(input_array, dtype=np.float32)

        if self.initialized:
            if mode == 'state':
                max_mem = min(self.mem_cntr, self.mem_size)
                input_max = np.max(self.state_memory[:max_mem], axis=0)
                input_min = np.min(self.state_memory[:max_mem], axis=0)
            else:
                input_max = float(self.action_size - 1)
                input_min = 0.0
        else:
            input_max = np.max(input_array, axis=0)
            input_min = np.min(input_array, axis=0)

        normalized_input = (input_array - input_min) / (input_max - input_min + 1e-10)
        normalized_input = np.clip(normalized_input, 0.0, 1.0)
        return normalized_input

    def max_denormalize(self, normalized_input, mode='state'):
        normalized_input = np.array(normalized_input, dtype=np.float32)

        if not self.initialized:
            raise ValueError("Denormalization requested on uninitialized buffer.")

        if mode == 'state':
            max_val = np.max(self.state_memory[:min(self.mem_cntr, self.mem_size)], axis=0)
            min_val = np.min(self.state_memory[:min(self.mem_cntr, self.mem_size)], axis=0)
        else:
            max_val = float(self.action_size - 1)
            min_val = 0.0

        denormal_input = normalized_input * (max_val - min_val) + min_val
        denormal_input = np.clip(denormal_input, min_val, max_val)
        return denormal_input

    def save_data(self, filepath):
        np.savez_compressed(
            filepath + ".npz",
            state_memory=self.state_memory[:min(self.mem_cntr, self.mem_size)],
            action_memory=self.action_memory[:min(self.mem_cntr, self.mem_size)],
            reward_memory=self.reward_memory[:min(self.mem_cntr, self.mem_size)],
            new_state_memory=self.new_state_memory[:min(self.mem_cntr, self.mem_size)],
            mem_cntr=self.mem_cntr
        )

    def load_data(self, filepath):
        if os.path.exists(filepath):
            data = np.load(filepath, allow_pickle=True)
            self.state_memory[:data['mem_cntr']] = data['state_memory']
            self.action_memory[:data['mem_cntr']] = data['action_memory']
            self.reward_memory[:data['mem_cntr']] = data['reward_memory']
            self.new_state_memory[:data['mem_cntr']] = data['new_state_memory']
            self.mem_cntr = data['mem_cntr']
            if self.mem_cntr > 1:
                self.initialized = True


# =============================================================================
# REVISED D3QN Agent with Exploration Tracking
# =============================================================================
class D3QNAgent:
    def __init__(self, state_size, action_size, load_model, discount_factor, epsilon,
                 epsilon_decay, epsilon_min, epsilon_start, epsilon_end, batch_size,
                 memsize, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        
        # CRITICAL FIX: Higher epsilon minimum for continued exploration
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = max(epsilon_min, 0.10)  # NEVER go below 10%
        
        self.batch_size = batch_size
        self.load_model = load_model
        self.learning_rate = learning_rate
        self.currentLoss = 0.0
        
        # NEW: Track action exploration
        self.action_counts = np.zeros(action_size)
        self.total_steps = 0

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        from keras.layers import Input
        from keras.models import Model
        
        # Fix Keras warning by using Input layer
        inputs = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(inputs)
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        outputs = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, add_exploration_bonus=True):
        """
        REVISED: Action selection with exploration bonus.
        
        Adds UCB-style bonus to Q-values for under-explored actions.
        """
        state = np.array([state], dtype=np.float32)
        q_value = self.model.predict(state, verbose=0)[0]
        
        # Add exploration bonus (UCB-style)
        if add_exploration_bonus and self.total_steps > 0:
            exploration_bonus = np.sqrt(np.log(self.total_steps + 1) / (self.action_counts + 1))
            exploration_weight = 0.05  # Small bonus to encourage exploration
            q_value = q_value + exploration_weight * exploration_bonus
        
        # Epsilon-greedy with floor
        if np.random.rand() <= max(self.epsilon, self.epsilon_min):
            action = random.randrange(self.action_size)
        else:
            action = int(np.argmax(q_value))
        
        # Update tracking
        self.action_counts[action] += 1
        self.total_steps += 1
        
        return action

    def train_model(self, memory):
        if memory is None or len(memory) < self.batch_size:
            return
        
        # REVISED: Respect epsilon minimum
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        next_states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        actions, rewards = [], []

        for i in range(self.batch_size):
            states[i] = memory[i][0]
            actions.append(memory[i][1])
            rewards.append(memory[i][2])
            next_states[i] = memory[i][3]

        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)

        target = self.model.predict(states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            a = actions[i]
            target[i][a] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        hist = self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.currentLoss = float(hist.history['loss'][0])
    
    def get_exploration_stats(self):
        """Get statistics about action exploration."""
        if self.total_steps == 0:
            return "No steps taken yet"
        
        visit_percentages = (self.action_counts / self.total_steps) * 100
        return f"Action visits: {visit_percentages}"


# =============================================================================
# State Helper Functions (matches server_mamfrl_2.py)
# =============================================================================
def get_thermal_agent_state(applications_data):
    """
    Compute thermal agent state from profiling data.
    Returns 4-element array: [avg_temp, avg_delta, mean_temps_after, mean_temps_delta]
    """
    if len(applications_data) == 0:
        return np.zeros(4, dtype=np.float32)

    temps_after_keys = ['CPU57_temp_after', 'Denver_temp_after', 'System_temp_after', 'GPU_temp_after', 'DDR_temp_after']
    temp_delta_keys = ['CPU57_temp_delta', 'Denver_temp_delta', 'System_temp_delta', 'GPU_temp_delta', 'DDR_temp_delta']

    avg_temp_after_vals = []
    avg_temp_delta_vals = []
    all_temps_after = []
    all_temp_deltas = []

    for ad in applications_data:
        avg_temp_after_vals.append(ad.get('avg_temp_after', 0.0) or 0.0)
        avg_temp_delta_vals.append(ad.get('avg_temp_delta', 0.0) or 0.0)
        for k in temps_after_keys:
            all_temps_after.append(ad.get(k, 0.0) or 0.0)
        for k in temp_delta_keys:
            all_temp_deltas.append(ad.get(k, 0.0) or 0.0)

    avg_temp = np.mean(avg_temp_after_vals) if avg_temp_after_vals else 0.0
    avg_delta = np.mean(avg_temp_delta_vals) if avg_temp_delta_vals else 0.0
    mean_temps_after = np.mean(all_temps_after) if all_temps_after else 0.0
    mean_temps_delta = np.mean(all_temp_deltas) if all_temp_deltas else 0.0

    return np.array([avg_temp, avg_delta, mean_temps_after, mean_temps_delta], dtype=np.float32)


def get_priority_agent_state(app_makespans, total_makespan):
    """
    Compute priority agent state.
    Returns (num_apps + 1)-element array: [total_makespan] + app_makespans
    """
    return np.array([total_makespan] + app_makespans, dtype=np.float32)


def get_profiler_agent_state(applications_data):
    """
    Get profiler agent state by parsing first application's data.
    Returns the full state vector from parse_state().
    """
    if not applications_data:
        return None
    parsed = parse_state(applications_data[0])
    if parsed is None:
        return None
    state, _, _, _, _, _, _, _, _ = parsed
    return state


# =============================================================================
# Main Training Function (MODEL-FREE)
# =============================================================================
def train_fixed_mamfrl_d3qn(
    client_socket,
    data_keys,
    experiment_time,
    clock_change_time,
    beta,
    load_model,
    learn_count,
    plan_count,  # Ignored in model-free version
    mem_size,
    learning_rate,
    discount_factor,
    epsilon,
    epsilon_decay,
    epsilon_min,
    epsilon_start,
    epsilon_end,
    reset_learning_rate_value,
    save_repetition,
    save_model,
    batch_size,
    agent_train_start,
    target_temp,
    server_name_1,
    server_name_2,
    server_name_main,
    profiling_data_list,
    application_profiles,
    applications_fixed,
    priority_combinations,
    frequency_combinations,
    num_cores_list
):
    """
    MODEL-FREE Multi-Agent RL training with D3QN agents.
    
    Key differences from MAMBRL:
    - NO environment model
    - NO planning or synthetic data generation
    - Only trains on real experiences
    - Simpler structure
    """
    
    logging.info("=" * 80)
    logging.info("Starting MAMFRL with D3QN Training (Model-Free)")
    logging.info("=" * 80)
    
    # Initialize performance tracker
    performance_tracker = BestPerformanceTracker()
    
    # Determine state size from first profiling
    temp_state = parse_state(profiling_data_list[0])
    if temp_state is None:
        raise ValueError("Failed to parse initial state")
    
    state_size = len(temp_state[0])
    num_apps = len(applications_fixed)

    # FIXED: Separate state sizes for each agent (matches server_mamfrl_2.py)
    # - Profiler uses full state from parse_state()
    # - Thermal uses 4-element state: [avg_temp, avg_delta, mean_temps_after, mean_temps_delta]
    # - Priority uses (num_apps + 1)-element state: [total_makespan] + app_makespans
    profiler_state_size = state_size
    thermal_state_size = 4
    priority_state_size = num_apps + 1

    # Action spaces
    profiler_action_space = len(frequency_combinations) * len(num_cores_list)
    # Thermal actions: (core_id, priority_change) for cores 1-5, changes [-1, 1]
    thermal_action_space = 10  # 5 cores * 2 changes
    priority_action_space = len(priority_combinations)

    logging.info(f"Profiler state size: {profiler_state_size}")
    logging.info(f"Thermal state size: {thermal_state_size}")
    logging.info(f"Priority state size: {priority_state_size}")
    logging.info(f"Profiler action space: {profiler_action_space}")
    logging.info(f"Thermal action space: {thermal_action_space}")
    logging.info(f"Priority action space: {priority_action_space}")

    # Initialize replay buffers with CORRECT state sizes
    profiler_memory = ReplayBuffer(max_size=mem_size, input_shape=profiler_state_size, action_size=profiler_action_space)
    thermal_memory = ReplayBuffer(max_size=mem_size, input_shape=thermal_state_size, action_size=thermal_action_space)
    priority_memory = ReplayBuffer(max_size=mem_size, input_shape=priority_state_size, action_size=priority_action_space)
    
    # Initialize D3QN agents
    profiler_agent = D3QNAgent(
        state_size=state_size,
        action_size=profiler_action_space,
        load_model=load_model,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        batch_size=batch_size,
        memsize=mem_size,
        learning_rate=learning_rate
    )
    
    thermal_agent = D3QNAgent(
        state_size=thermal_state_size,  # FIXED: Use thermal_state_size (4)
        action_size=thermal_action_space,
        load_model=load_model,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        batch_size=batch_size,
        memsize=mem_size,
        learning_rate=learning_rate
    )

    priority_agent = D3QNAgent(
        state_size=priority_state_size,  # FIXED: Use priority_state_size (num_apps + 1)
        action_size=priority_action_space,
        load_model=load_model,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        batch_size=batch_size,
        memsize=mem_size,
        learning_rate=learning_rate
    )
    
    # Load models if requested
    if load_model:
        model_dir = save_dir
        profiler_memory.load_data(os.path.join(model_dir, 'mamfrl_profiler_state_transitions.csv.npz'))
        thermal_memory.load_data(os.path.join(model_dir, 'mamfrl_thermal_state_transitions.csv.npz'))
        priority_memory.load_data(os.path.join(model_dir, 'mamfrl_priority_state_transitions.csv.npz'))

        if os.path.exists(os.path.join(model_dir, "mamfrl_profiler_model_data.weights.h5")):
            profiler_agent.model.load_weights(os.path.join(model_dir, "mamfrl_profiler_model_data.weights.h5"))
        if os.path.exists(os.path.join(model_dir, "mamfrl_thermal_model_data.weights.h5")):
            thermal_agent.model.load_weights(os.path.join(model_dir, "mamfrl_thermal_model_data.weights.h5"))
        if os.path.exists(os.path.join(model_dir, "mamfrl_priority_model_data.weights.h5")):
            priority_agent.model.load_weights(os.path.join(model_dir, "mamfrl_priority_model_data.weights.h5"))

        logging.info(f"Models loaded: profiler={profiler_memory.mem_cntr}, "
                    f"thermal={thermal_memory.mem_cntr}, priority={priority_memory.mem_cntr}")
    
    # Tracking data structures
    ts = []
    makespan_per_app = [[] for _ in applications_fixed]
    priority_per_app = [[] for _ in applications_fixed]
    energy_per_app = [[] for _ in applications_fixed]
    temperature_per_app = [[] for _ in applications_fixed]
    qmax_per_app = [[] for _ in applications_fixed]
    loss_per_app = [[] for _ in applications_fixed]
    reward_per_app = [[] for _ in applications_fixed]
    frequency_per_app = [[] for _ in applications_fixed]
    cores_per_app = [[] for _ in applications_fixed]
    
    # Current state tracking - use application IDs as keys (hashable)
    app_id_list = [app.get('id', i+1) for i, app in enumerate(applications_fixed)]
    app_variant_list = [app.get('variant', f'app_{i+1}') for i, app in enumerate(applications_fixed)]

    current_states = {app_id: np.zeros(state_size, dtype=np.float32) for app_id in app_id_list}
    previous_temperatures = {app_id: 30.0 for app_id in app_id_list}

    # Aggregated tracking for plotting (matches server_mamfrl_2.py)
    total_makespan_data = []
    total_energy_data = []
    total_branch_misses_data = []
    total_cache_misses_data = []
    profiler_qmax_data = []
    thermal_qmax_data = []
    priority_qmax_data = []
    profiler_loss_data = []
    thermal_loss_data = []
    priority_loss_data = []
    profiler_rewards = []
    thermal_rewards = []
    priority_rewards = []
    total_rewards = []
    priority_data = []
    thermal_data = []  # Added for proper temperature tracking
    num_cores_data = []  # Added for standardized plotting
    cores_combination_data = []  # Bitmask for plotting
    cores_str_data = []  # String representation for CSV
    freq_data = []
    priority_values_data = []

    # CSV data collection
    collected_data = []

    # Use server_name_main if provided (from server_combined.py), otherwise generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if server_name_main:
        csv_filename = server_name_main if os.path.isabs(server_name_main) else os.path.join(save_dir, server_name_main)
    else:
        csv_filename = os.path.join(save_dir, f"MAMFRL_D3QN_{timestamp}_{experiment_time}ep_beta{beta}_lr{learning_rate}_eps{epsilon_min}_batch{batch_size}.csv")

    iterations_since_last_save = 0
    checkpoint_interval = save_repetition

    # =========================================================================
    # Initialize Online Plotting using unified OnlinePlotter
    # =========================================================================
    plotter = OnlinePlotter(
        data_keys=data_keys,
        experiment_time=experiment_time,
        module_name="MAMFRL_D3QN",
        save_dir=save_dir,
        server_name_1=server_name_1,
        server_name_2=server_name_2,
        baseline_label="MAMBRL (Model-Based)",
        previous_label="MAMFRL (Previous)"
    )

    # Historical data loading now handled by OnlinePlotter

    # =========================================================================
    # FIXED TRAINING LOOP - Matches server_mamfrl_2.py communication pattern
    # Pattern: 1) Choose actions, 2) Build app instances, 3) SEND to client,
    #          4) RECEIVE results, 5) Process rewards, 6) Train
    # =========================================================================

    # CoreList for core allocation (TX2 has 6 cores, 0-5, typically use 1-5)
    CoreList = [1, 2, 3, 4, 5]

    # Priority queue for thermal-based core management
    class PriorityQueue:
        def __init__(self):
            self.core_priorities = {}
            self.prioritized_cores_set = set()

        def update_core_priority(self, core_id, priority_change):
            self.core_priorities[core_id] = self.core_priorities.get(core_id, 0) + priority_change
            if self.core_priorities[core_id] < 0:
                self.core_priorities[core_id] = 0
            self.prioritized_cores_set = {k for k, v in self.core_priorities.items() if v > 0}

    def generate_set_of_cores(num_cores, prioritized_cores_set, CoreList):
        """Generate core allocation considering thermal priorities."""
        if len(prioritized_cores_set) >= num_cores:
            return random.sample(list(prioritized_cores_set), num_cores)
        else:
            needed = num_cores - len(prioritized_cores_set)
            extra_cores = [c for c in CoreList if c not in prioritized_cores_set]
            if len(extra_cores) < needed:
                needed = len(extra_cores)
            return list(prioritized_cores_set) + random.sample(extra_cores, needed)

    priority_queue = PriorityQueue()

    # Thermal actions: (core_id, priority_change) - same as server_mamfrl_2.py
    thermal_actions = []
    for core_id in range(1, 6):
        for pc in [-1, 1]:
            thermal_actions.append((core_id, pc))

    # Initialize states using helper functions (matches server_mamfrl_2.py)
    profiler_state = get_profiler_agent_state(profiling_data_list)
    if profiler_state is None:
        logging.error("Could not parse initial profiler state.")
        return None

    makespans_init = [pd_.get('time_elapsed', 0.0) for pd_ in profiling_data_list]
    total_makespan_init = sum(makespans_init) if makespans_init else 0.0
    priority_state = get_priority_agent_state(makespans_init, total_makespan_init)
    thermal_state = get_thermal_agent_state(profiling_data_list)

    avg_temperature_prev = target_temp

    logging.info("Starting main training loop (FIXED - server_mamfrl_2.py pattern)...")

    for t in range(experiment_time):
        logging.info(f"\n{'='*60}")
        logging.info(f"MAMFRL Iteration {t + 1}/{experiment_time}")
        logging.info(f"{'='*60}")

        # Check if stuck and needs exploration boost
        if performance_tracker.needs_exploration_boost(threshold=20):
            logging.warning(f"No improvement for {performance_tracker.iterations_since_improvement} iterations!")
            profiler_agent.epsilon = min(profiler_agent.epsilon * 1.5, 0.5)
            thermal_agent.epsilon = min(thermal_agent.epsilon * 1.5, 0.5)
            priority_agent.epsilon = min(priority_agent.epsilon * 1.5, 0.5)
            logging.info(f"Exploration boost: epsilon={profiler_agent.epsilon:.3f}")

        # -------------------------------------------------------------
        # 1) Get actions from agents (MODEL-FREE: using correct state sizes)
        # -------------------------------------------------------------
        profiler_action = profiler_agent.get_action(profiler_state)
        freq_step = frequency_combinations[profiler_action % len(frequency_combinations)][0]
        num_cores = num_cores_list[profiler_action // len(frequency_combinations) % len(num_cores_list)]

        # Thermal agent uses thermal_state (4 elements)
        thermal_action = thermal_agent.get_action(thermal_state)
        core_id_change, priority_change = thermal_actions[thermal_action % len(thermal_actions)]
        priority_queue.update_core_priority(core_id_change, priority_change)

        # Priority agent uses priority_state (num_apps + 1 elements)
        priority_action = priority_agent.get_action(priority_state)
        current_priority_tuple = priority_combinations[priority_action % len(priority_combinations)]

        chosen_freq_comb = [freq_step] * 5

        # -------------------------------------------------------------
        # 2) Build application instances with proper format for client
        # -------------------------------------------------------------
        applications_run = []
        available_cores = generate_set_of_cores(num_cores, priority_queue.prioritized_cores_set, CoreList)
        initial_available_cores = available_cores.copy()

        app_with_priority = [(current_priority_tuple[idx % len(current_priority_tuple)], applications_fixed[idx])
                           for idx in range(len(applications_fixed))]
        app_with_priority_sorted = sorted(app_with_priority, key=lambda x: x[0], reverse=True)

        for priority_val, app in app_with_priority_sorted:
            app_id = app['id']
            # Use str() for consistency with profiling phase key format
            app_input_arg = str(app.get('app_args', app.get('input_arg', '')))
            profile_key = (app_id, app_input_arg)
            app_profile = application_profiles.get(profile_key) if application_profiles else None

            level_of_parallelism = 1.0
            if app_profile:
                level_of_parallelism = app_profile.get('parallelism_level', 1.0) or 1.0

            num_cores_to_allocate = max(int(level_of_parallelism), 1)
            num_cores_to_allocate = min(num_cores, num_cores_to_allocate)

            if len(available_cores) >= num_cores_to_allocate:
                cores_assigned = random.sample(available_cores, num_cores_to_allocate)
                for core in cores_assigned:
                    available_cores.remove(core)
            else:
                cores_assigned = available_cores.copy()
                available_cores.clear()
                remaining = [c for c in initial_available_cores if c not in cores_assigned]
                needed = num_cores_to_allocate - len(cores_assigned)
                if remaining and needed > 0:
                    cores_assigned.extend(random.sample(remaining, min(len(remaining), needed)))

            cores_assigned = list(set(cores_assigned))
            cores_assigned_str = ','.join(map(str, cores_assigned))
            frequencies_assigned = chosen_freq_comb[:len(cores_assigned)]

            # FIXED: Include ALL fields that client_evaluate_tx2.py expects
            app_instance = {
                'id': app['id'],
                'benchmark': app['benchmark'],
                'variant': app['variant'],
                'input_arg': app['input_arg'],
                'app_args': app.get('app_args', app.get('input_arg', '')),  # CRITICAL for client
                'priority': priority_val,
                'frequencies': frequencies_assigned,
                'num_cores': len(cores_assigned),
                'cores': cores_assigned_str,
                'action': 'run'
            }
            applications_run.append(app_instance)

        logging.info(f"Actions: freq={freq_step}, cores={num_cores}, priority={current_priority_tuple}")

        # -------------------------------------------------------------
        # 3) SEND applications to client (FIXED pattern)
        # -------------------------------------------------------------
        send_msg_dict = {
            'applications': applications_run,
            'run_mode': 'parallel'
        }
        send_msg = json.dumps(send_msg_dict)
        client_socket.send((send_msg + '\n').encode())
        logging.info(f"Sent {len(applications_run)} applications to client")

        # -------------------------------------------------------------
        # 4) RECEIVE results from client (simple buffer pattern)
        # -------------------------------------------------------------
        data_received = False
        recv_buffer = ''
        start_time_wait = time.time()
        timeout = 600
        new_profiling_data_list = []

        while not data_received and time.time() - start_time_wait < timeout:
            try:
                data = client_socket.recv(4096)
                if data:
                    recv_buffer += data.decode()
                    while '\n' in recv_buffer:
                        msg, recv_buffer = recv_buffer.split('\n', 1)
                        try:
                            msg_json = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        pd_list = msg_json.get('profiling_data_list', [])
                        if len(pd_list) == len(applications_run):
                            data_received = True
                            new_profiling_data_list = pd_list
                            # Debug: Print received profiling data like evaluation server
                            print("\n=== Profiling data received from client ===")
                            print(json.dumps(pd_list, indent=2, sort_keys=True, default=str))
                            print("===========================================\n")
                            send_ack = json.dumps({'status': 'received'})
                            client_socket.send((send_ack + '\n').encode())
                        else:
                            continue
                else:
                    time.sleep(1)
            except socket.timeout:
                logging.error("Socket timeout")
                break
            except Exception as e:
                logging.error(f"Error receiving: {e}")
                break

        if not data_received:
            logging.error("Timeout waiting for data")
            continue

        # -------------------------------------------------------------
        # 5) Process results and calculate rewards
        # -------------------------------------------------------------
        profiling_data_list_current = new_profiling_data_list

        # Normalize profiling data to compute avg_temp_after from thermal zones
        profiling_data_list_current = [normalize_profiling_data(d) for d in profiling_data_list_current]

        makespans = [d.get('time_elapsed', 0.0) for d in profiling_data_list_current]
        total_makespan = sum(makespans)
        energies = [d.get('total_energy_consumption', 0.0) for d in profiling_data_list_current]
        total_energy = sum(energies)

        # Get temperature with fallbacks
        temperatures = [get_temperature_with_fallbacks(d, target_temp) for d in profiling_data_list_current]
        avg_temperature = float(np.mean(temperatures)) if temperatures else float(target_temp)

        # Compute union of core combinations (both bitmask and string)
        union_cores = 0
        all_cores_set = set()
        for d in profiling_data_list_current:
            cores_str = d.get('cores', '')
            union_cores |= subset_to_int(cores_str)
            if cores_str:
                for c in cores_str.split(','):
                    c = c.strip()
                    if c.isdigit():
                        all_cores_set.add(int(c))
        cores_combination_data.append(union_cores)
        cores_str_data.append(','.join(map(str, sorted(all_cores_set))) if all_cores_set else '')
        freq_data.append(freq_step)

        # Get target metrics from profiling
        target_makespans = [d.get('makespan_all_cores_frequency_11', 0.0) or 0.0 for d in profiling_data_list_current]
        total_target_makespan = sum(target_makespans) if any(target_makespans) else total_makespan
        target_energies = [d.get('energy_all_cores_frequency_0', 0.0) or 0.0 for d in profiling_data_list_current]
        total_target_energy = sum(target_energies) if any(target_energies) else total_energy

        logging.info(f"Results: makespan={total_makespan:.3f}s, energy={total_energy:.2f}J, temp={avg_temperature:.1f}C")

        # Calculate rewards (MODEL-FREE: use adaptive rewards)
        profiler_reward, got_improvement = get_reward_profiler(
            total_makespan, total_energy, total_target_makespan, total_target_energy,
            beta=beta, best_makespan=performance_tracker.best_makespan, iteration=t
        )
        thermal_reward = get_reward_thermal(avg_temperature, avg_temperature_prev, target_temp)
        priority_reward = get_reward_priority(total_makespan, total_target_makespan)
        total_reward = profiler_reward + 0.2 * thermal_reward

        logging.info(f"Rewards: profiler={profiler_reward:.4f}, thermal={thermal_reward:.4f}, total={total_reward:.4f}")

        # Update performance tracker
        improved = performance_tracker.update(total_makespan, total_energy, avg_temperature)

        # Track metrics
        for app_idx, d in enumerate(profiling_data_list_current):
            if app_idx < len(makespan_per_app):
                makespan_per_app[app_idx].append(d.get('time_elapsed', 0.0))
                energy_per_app[app_idx].append(d.get('total_energy_consumption', 0.0))
                temperature_per_app[app_idx].append(get_temperature_with_fallbacks(d, target_temp))
                priority_per_app[app_idx].append(current_priority_tuple[app_idx % len(current_priority_tuple)])
                frequency_per_app[app_idx].append(freq_step)
                cores_per_app[app_idx].append(num_cores)

        ts.append(t)

        # Aggregated data collection for plotting
        total_makespan_data.append(total_makespan)
        total_energy_data.append(total_energy)
        # Handle None values for branch/cache misses
        total_branch_misses = sum(d.get('branch_misses', 0) or 0 for d in profiling_data_list_current)
        total_cache_misses = sum(d.get('cache_misses', 0) or 0 for d in profiling_data_list_current)
        total_branch_misses_data.append(total_branch_misses)
        total_cache_misses_data.append(total_cache_misses)
        thermal_data.append(avg_temperature)  # Accumulate temperature history
        num_cores_data.append(num_cores)  # Track number of cores for standardized plotting
        profiler_rewards.append(profiler_reward)
        thermal_rewards.append(thermal_reward)
        priority_rewards.append(priority_reward)
        total_rewards.append(total_reward)
        priority_data.append(priority_tuple_to_int(current_priority_tuple))
        priority_values_data.append(np.mean(current_priority_tuple))

        logging.info(f"[MAMFRL iter={t}] Temp={avg_temperature:.1f}C, BranchMiss={total_branch_misses}, CacheMiss={total_cache_misses}")

        # Get new states using helper functions (matches server_mamfrl_2.py)
        makespans_new = [pd_.get('time_elapsed', 0.0) for pd_ in profiling_data_list_current]
        total_mk_new = sum(makespans_new)
        new_priority_state = get_priority_agent_state(makespans_new, total_mk_new)
        new_thermal_state = get_thermal_agent_state(profiling_data_list_current)
        new_profiler_state = get_profiler_agent_state(profiling_data_list_current)
        if new_profiler_state is None:
            new_profiler_state = profiler_state.copy()

        # -------------------------------------------------------------
        # 6) Store experiences and train (MODEL-FREE: only real buffer)
        # -------------------------------------------------------------
        profiler_memory.store_observations(profiler_state, profiler_action, profiler_reward, new_profiler_state)
        thermal_memory.store_observations(thermal_state, thermal_action, thermal_reward, new_thermal_state)
        priority_memory.store_observations(priority_state, priority_action, priority_reward, new_priority_state)

        # Train agents (MODEL-FREE: only real buffer, no generative data)
        if profiler_memory.mem_cntr >= agent_train_start:
            profiler_batch = profiler_memory.sample_observation(batch_size, random=True)
            thermal_batch = thermal_memory.sample_observation(batch_size, random=True)
            priority_batch = priority_memory.sample_observation(batch_size, random=True)

            if profiler_batch:
                profiler_agent.train_model(profiler_batch)
            if thermal_batch:
                thermal_agent.train_model(thermal_batch)
            if priority_batch:
                priority_agent.train_model(priority_batch)

            profiler_agent.update_target_model()
            thermal_agent.update_target_model()
            priority_agent.update_target_model()

        # Update states for next iteration
        profiler_state = new_profiler_state
        thermal_state = new_thermal_state
        priority_state = new_priority_state
        avg_temperature_prev = avg_temperature

        # Q-values and loss tracking for all agents
        p_qval = profiler_agent.model.predict(np.array([new_profiler_state]), verbose=0)[0]
        t_qval = thermal_agent.model.predict(np.array([new_thermal_state]), verbose=0)[0]
        pr_qval = priority_agent.model.predict(np.array([new_priority_state]), verbose=0)[0]

        profiler_qmax_data.append(np.max(p_qval))
        thermal_qmax_data.append(np.max(t_qval))
        priority_qmax_data.append(np.max(pr_qval))
        profiler_loss_data.append(profiler_agent.currentLoss)
        thermal_loss_data.append(thermal_agent.currentLoss)
        priority_loss_data.append(priority_agent.currentLoss)

        for app_idx in range(len(qmax_per_app)):
            qmax_per_app[app_idx].append(np.max(p_qval))
            loss_per_app[app_idx].append(profiler_agent.currentLoss)
            reward_per_app[app_idx].append(total_reward)

        # =====================================================================
        # Online Plotting - Update comparison plots using unified plotter
        # =====================================================================
        # Prepare for plotting using unified create_data_map
        current_run_data_map = create_data_map(
            total_makespan_data=total_makespan_data,
            num_cores_data=num_cores_data,
            cores_combination_data=cores_combination_data,
            profiler_qmax_data=profiler_qmax_data,
            total_energy_data=total_energy_data,
            freq_data=freq_data,
            priority_data=priority_data,
            thermal_data=thermal_data,
            total_branch_misses_data=total_branch_misses_data,
            total_cache_misses_data=total_cache_misses_data,
            priority_values_data=priority_values_data,
            total_rewards=total_rewards,
            thermal_qmax_data=thermal_qmax_data,
            priority_qmax_data=priority_qmax_data
        )

        # Update plots using unified OnlinePlotter
        plotter.update(current_run_data_map, t)

        # Collect data for CSV
        row_data = {
            'time': t,
            'makespan': total_makespan,
            'priority': np.mean([p[-1] for p in priority_per_app if p]),
            'energy': total_energy,
            'qmax': max(profiler_qmax_data[-1], thermal_qmax_data[-1], priority_qmax_data[-1]),
            'loss': (profiler_loss_data[-1] + thermal_loss_data[-1] + priority_loss_data[-1]) / 3.0,
            'freq': freq_step,
            'cores': num_cores,
            'thermal': avg_temperature,
            'reward': total_reward,
            'branchmisses': total_branch_misses_data[-1],
            'cachemisses': total_cache_misses_data[-1],
            'priority_combination': priority_data[-1],
        }

        # Preserve ALL raw fields from profiling data (thermal_zone*, energy_*_j, power_*_w, etc.)
        # Following the pattern from server_evaluate_tx2.py to save complete profiling data
        for app_idx, d in enumerate(profiling_data_list):
            prefix = f"app{app_idx}_"
            for k, v in d.items():
                # Add all raw fields with app prefix to avoid collisions
                col_name = f"{prefix}{k}"
                if col_name not in row_data:
                    row_data[col_name] = v

        # Also save aggregated thermal zone data across all apps
        if profiling_data_list:
            # Get thermal zones from first app (they should be system-wide)
            first_d = profiling_data_list[0]
            for k, v in first_d.items():
                if k.startswith('thermal_zone') or k.startswith('energy_') or k.startswith('power_'):
                    if k not in row_data:
                        row_data[k] = v

        collected_data.append(row_data)

        # Periodic saving
        iterations_since_last_save += 1
        if iterations_since_last_save >= checkpoint_interval and checkpoint_interval != 0:
            if save_model:
                model_dir = save_dir
                os.makedirs(model_dir, exist_ok=True)
                profiler_agent.model.save_weights(os.path.join(model_dir, "mamfrl_profiler_model_data.weights.h5"))
                thermal_agent.model.save_weights(os.path.join(model_dir, "mamfrl_thermal_model_data.weights.h5"))
                priority_agent.model.save_weights(os.path.join(model_dir, "mamfrl_priority_model_data.weights.h5"))
                profiler_memory.save_data(os.path.join(model_dir, 'mamfrl_profiler_state_transitions.csv'))
                thermal_memory.save_data(os.path.join(model_dir, 'mamfrl_thermal_state_transitions.csv'))
                priority_memory.save_data(os.path.join(model_dir, 'mamfrl_priority_state_transitions.csv'))
                logging.info("[Save models and replay memories]")
            if collected_data:
                df_csv = pd.DataFrame(collected_data)
                df_csv.to_csv(csv_filename, index=False)
                logging.info(f"Saved collected data to '{csv_filename}'")
            iterations_since_last_save = 0

        # Progress logging
        if t % 5 == 0:
            stats = performance_tracker.get_stats()
            logging.info(f"Progress: best_makespan={stats['best_makespan']:.3f}s, improvements={stats['total_improvements']}")

    # Save final results - using collected_data format (matches server_mamfrl_2.py)
    if collected_data and csv_filename:
        try:
            df_csv = pd.DataFrame(collected_data)
            df_csv.to_csv(csv_filename, index=False)
            logging.info(f"Saved collected data to '{csv_filename}'")
        except Exception as e:
            logging.error(f"Failed to save CSV: {e}", exc_info=True)

    # Save final plot using unified plotter - derive from csv_filename for consistency
    if csv_filename:
        plot_filename = os.path.basename(csv_filename).replace('.csv', '_plot.png')
        plotter.save(os.path.join(save_dir, plot_filename))

    plotter.close()  # Free memory

    logging.info("=" * 80)
    logging.info("MAMFRL with D3QN Training Complete (Model-Free)")
    if early_stopped:
        logging.info(f"Training stopped early after reaching convergence")
        logging.info(f"Early stopping best makespan: {best_makespan:.3f}s")
    logging.info(f"Final best makespan: {performance_tracker.best_makespan:.3f}s")
    logging.info(f"Total improvements: {len(performance_tracker.improvement_history)}")
    logging.info("=" * 80)

    return ts, makespan_per_app, energy_per_app, temperature_per_app


if __name__ == '__main__':
    logging.info("MAMFRL D3QN (Model-Free) script loaded")
    logging.info("This script should be called from combined_tx2.py")