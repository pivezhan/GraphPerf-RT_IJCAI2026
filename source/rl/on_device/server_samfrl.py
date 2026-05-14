#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAMFRL - Single-Agent Model-Free RL for Jetson TX2
FIXED VERSION - Compatible with client_evaluate_tx2.py and combined_tx2_fixed.py

KEY CHANGES FROM server_samfrl.py:
===================================
1. Function renamed to train_samfrl_fixed() for consistency
2. Filename format: SAMFRL_YYYYMMDD_HHMMSS_EXPep_betaB_lrL_epsE_batchB.csv
3. Added helper functions: parse_hyperparams_from_filename(), load_historical_data(), extract_server_name()
4. Historical data loading with cross-module comparison (SAMFRL vs MAMFRL/MAMBRL)
5. Data saving: One row per iteration (aggregated per-app data)
6. Online plotting with historical comparison and X-axis synchronization
7. Return values: (ts, makespan_per_app, energy_per_app, temperature_per_app)

SAMFRL CHARACTERISTICS:
=======================
- Single-Agent: Only Profiler agent (no Thermal, no Priority agents)
- Model-Free: No environment model, no planning, only real experiences
- Action space: Frequency + Core allocation combinations
- Fixed priority: All apps get priority=10
- Fixed thermal action: No thermal adjustments

CLIENT MESSAGE FORMAT (client_evaluate_tx2.py expects):
======================================================
{
    "id": int,                 # Application identifier
    "benchmark": str,          # e.g., "fft", "gemm"
    "variant": str,            # e.g., "bin-omp-tasks" for BOTS
    "input_arg": str,          # e.g., "5" for BOTS
    "app_args": str,           # CRITICAL: same as input_arg
    "frequencies": list,       # List of frequency indices per core
    "num_cores": int,          # Number of cores allocated
    "cores": str,              # Comma-separated core list (e.g., "1,2,3")
    "priority": int,           # RT priority value (fixed at 10)
    "action": str              # "profile" or "run"
}
"""

import logging
import json
import time
import numpy as np
import random
import csv
import os
from datetime import datetime
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')  # non-interactive backend
# Configure matplotlib to reduce memory usage
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 1000
import matplotlib.pyplot as plt
import pandas as pd
import re
import socket
import threading
import argparse

# Import unified plotting module
from live_plotter import (
    OnlinePlotter, create_data_map,
    load_historical_data, extract_server_name, parse_hyperparams_from_filename
)
from server_combined import DATA_KEYS

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input
from keras.models import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of this script for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "save_model")

STATE_SIZE = None

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


def priority_tuple_to_int(tup):
    """Convert priority tuple to integer for logging."""
    return int(''.join(str(x) for x in tup))


# =============================================================================
# Historical Data Loading and Plotting Utilities
# =============================================================================
def parse_hyperparams_from_filename(filename):
    """
    Parse hyperparameters from a filename.
    Returns dict with keys: exp, beta, lr, eps_min, batch, module_name
    Returns None if parsing fails.

    Supports filenames like:
    - SAMFRL_20251204_155454_100ep_beta1_lr0.05_eps0.1_batch32.csv (new format)
    - SAMFRL_20251204_155454_100ep_beta1.csv (old format)
    """
    basename = os.path.basename(filename)

    # Try new format: MODULE_YYYYMMDD_HHMMSS_EXPep_betaB_lrL_epsE_batchB.csv
    # FIXED: Use (.+?) not (\w+) to match module names with underscores
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
    # FIXED: Use (.+?) not (\w+) to match module names with underscores
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
                    (for cross-module comparison where MAMBRL/MAMFRL has 3x rows)

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
                # e.g., 300 rows -> 100 rows (MAMFRL has 3 rows per iteration for 3 apps)
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

    Returns descriptive names like 'SAMFRL', 'MAMFRL', 'MAMBRL'
    """
    if not tuning_name:
        return "Unknown"

    basename = os.path.basename(tuning_name).upper()

    # Check for known module names
    if "SAMFRL" in basename:
        return "SAMFRL (Single-Agent)"
    elif "MAMBRL" in basename:
        return "MAMBRL (Model-Based)"
    elif "MAMFRL" in basename:
        return "MAMFRL (Model-Free)"

    # Fallback: try old pattern
    match = re.search(r"server_(.+?)_\d+", tuning_name)
    if match:
        return match.group(1)

    # Last resort: use filename prefix
    if '_' in basename:
        return basename.split('_')[0]

    return "Historical"


# =============================================================================
# State Parsing
# =============================================================================
def parse_state(profiling_data):
    """
    Parse profiling data dict into a state vector.

    Returns:
        state: np.array, the state vector for the profiler agent
        total_energy_consumption: float
        avg_temp_after: float
        time_elapsed: float
        branch_misses: int
        cache_misses: int
        target_makespan: float
        target_energy: float
        parallelism_level: float
    """
    global STATE_SIZE
    try:
        freq_list = profiling_data.get('frequencies', [0])
        c_c = freq_list[0] if freq_list else 0
        subset_str = profiling_data.get('cores', '')
        subset_int = subset_to_int(subset_str)
        utilization = profiling_data.get('utilization', 0.0)
        time_elapsed = profiling_data.get('time_elapsed', 0.0)
        avg_temp_after = profiling_data.get('avg_temp_after', 0.0)
        avg_temp_delta = profiling_data.get('avg_temp_delta', 0.0)
        total_energy_consumption = profiling_data.get('total_energy_consumption', 0.0)

        target_makespan = profiling_data.get('makespan_all_cores_frequency_11', 0.0)
        target_energy = profiling_data.get('energy_all_cores_frequency_0', 0.0)
        parallelism_level = profiling_data.get('parallelism_level', 0.0)

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

        energy_consumption = []
        for key in ['CPU57_energy_joules', 'Denver_energy_joules', 'System_energy_joules', 'GPU_energy_joules', 'DDR_energy_joules']:
            energy_consumption.append(profiling_data.get(key, 0.0))

        temps_after = []
        for key in ['CPU57_temp_after', 'Denver_temp_after', 'System_temp_after', 'GPU_temp_after', 'DDR_temp_after']:
            temps_after.append(profiling_data.get(key, 0.0))

        temp_deltas = []
        for key in ['CPU57_temp_delta', 'Denver_temp_delta', 'System_temp_delta', 'GPU_temp_delta', 'DDR_temp_delta']:
            temp_deltas.append(profiling_data.get(key, 0.0))

        state = np.array([
            c_c,
            subset_int,
            utilization,
            cycles,
            cache_references,
            cache_misses,
            branch_instructions,
            task_clock,
            context_switches,
            minor_faults,
            major_faults,
            branch_misses,
            branches,
            instructions,
            page_faults,
            cpu_clock,
            time_elapsed,
            *energy_consumption,
            *temps_after,
            *temp_deltas,
            avg_temp_delta
        ], dtype=np.float32)

        if STATE_SIZE is None:
            STATE_SIZE = len(state)
            logging.info(f"Determined state size: {STATE_SIZE}")

        return state, total_energy_consumption, avg_temp_after, time_elapsed, branch_misses, cache_misses, target_makespan, target_energy, parallelism_level
    except Exception as e:
        logging.warning(f"Error parsing state: {e}")
        return None


# =============================================================================
# Reward Function
# =============================================================================
def get_reward_profiler(makespan, avg_energy_consumption, target_makespan, target_energy, beta=0.5):
    """
    Profiler reward function (simple ratio-based).
    Invert the ratios so that lower makespan/energy gives higher reward.
    """
    epsilon = 1e-3
    if makespan <= epsilon:
        makespan = target_makespan
    if avg_energy_consumption <= epsilon:
        avg_energy_consumption = target_energy
    makespan_reward = target_makespan / (makespan + epsilon)
    energy_reward = target_energy / (avg_energy_consumption + epsilon)
    reward_value = beta * makespan_reward + (1 - beta) * energy_reward
    logging.debug(f"Target makespan was {target_makespan} and makespan was {makespan}. Target energy was {target_energy} and energy was {avg_energy_consumption}. Makespan reward: {makespan_reward}, Energy reward: {energy_reward}, Combined reward: {reward_value}")
    return reward_value


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

    def sample_observation(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        if max_mem < batch_size:
            return None
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return list(zip(
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch]
        ))

    def max_normalize(self, input_array, mode='state'):
        input_array = np.array(input_array)
        if self.initialized:
            if mode == 'state':
                input_max = np.max(self.state_memory[:min(self.mem_cntr, self.mem_size)], axis=0)
                input_min = np.min(self.state_memory[:min(self.mem_cntr, self.mem_size)], axis=0)
            else:
                input_max = self.action_size
                input_min = 0
        else:
            input_max = np.max(input_array, axis=0)
            input_min = np.min(input_array, axis=0)

        normalized_input = (input_array - input_min) / (input_max - input_min + 1e-10)
        normalized_input = np.clip(normalized_input, 0.0, 1.0)
        return normalized_input

    def max_denormalize(self, normalized_input, mode='state'):
        if not self.initialized:
            raise ValueError("Denormalization requested on uninitialized buffer.")
        if mode == 'state':
            max_val = np.max(self.state_memory[:min(self.mem_cntr, self.mem_size)], axis=0)
            min_val = np.min(self.state_memory[:min(self.mem_cntr, self.mem_size)], axis=0)
        else:
            max_val = self.action_size - 1
            min_val = 0

        denormal_input = normalized_input * (max_val - min_val) + min_val
        denormal_input = np.clip(denormal_input, min_val, max_val)
        return denormal_input

    def save_data(self, filepath):
        np.savez_compressed(filepath + ".npz",
                            state_memory=self.state_memory[:min(self.mem_cntr,self.mem_size)],
                            action_memory=self.action_memory[:min(self.mem_cntr,self.mem_size)],
                            reward_memory=self.reward_memory[:min(self.mem_cntr,self.mem_size)],
                            new_state_memory=self.new_state_memory[:min(self.mem_cntr,self.mem_size)],
                            mem_cntr=self.mem_cntr)

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
# DQN Agent
# =============================================================================
class DQNAgent:
    def __init__(self, state_size, action_size, load_model, discount_factor, epsilon, epsilon_decay, epsilon_min, epsilon_start, epsilon_end, batch_size, memsize, learning_rate, memory=None):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.load_model = load_model
        self.learning_rate = learning_rate
        self.memory = memory

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.currentLoss = 0

    def build_model(self):
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

    def get_action(self, state):
        state = np.array([state], dtype=np.float32)
        state_norm = self.memory.max_normalize(state, mode='state') if self.memory and self.memory.initialized else state
        q_value = self.model.predict(state_norm, verbose=0)[0]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(q_value)

    def train_model(self, memory):
        if memory is None or len(memory) < self.batch_size:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        next_states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        actions, rewards = [], []
        for i in range(self.batch_size):
            states[i] = memory[i][0]
            actions.append(memory[i][1])
            rewards.append(memory[i][2])
            next_states[i] = memory[i][3]

        states_norm = self.memory.max_normalize(states, mode='state') if self.memory and self.memory.initialized else states
        next_states_norm = self.memory.max_normalize(next_states, mode='state') if self.memory and self.memory.initialized else next_states

        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        target = self.model.predict(states_norm, verbose=0)
        target_val = self.target_model.predict(next_states_norm, verbose=0)

        for i in range(self.batch_size):
            a = actions[i]
            target[i][a] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        hist = self.model.fit(states_norm, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.currentLoss = hist.history['loss'][0]


# =============================================================================
# Helper Functions for State Construction
# =============================================================================
def get_profiler_agent_state(applications_data):
    """Get profiler agent state by parsing first application's data."""
    if not applications_data:
        return None
    parsed = parse_state(applications_data[0])
    if parsed is None:
        return None
    state, _, _, _, _, _, _, _, _ = parsed
    return state


# =============================================================================
# Main Training Function - SAMFRL FIXED
# =============================================================================
def train_fixed_samfrl(client_socket, data_keys=DATA_KEYS, experiment_time=100, clock_change_time=30, beta=1.0,
                 load_model=False, learn_count=100, plan_count=100, mem_size=100000, learning_rate=0.1,
                 discount_factor=0.99, epsilon=1, epsilon_decay=0.95, epsilon_min=0.10, epsilon_start=1.0,
                 epsilon_end=0.01, reset_learning_rate_value=20, save_repetition=50, save_model=True,
                 batch_size=64, agent_train_start=64, target_temp=50,
                 server_name_1='', server_name_2='', server_name_main='samfrl_main.csv',
                 profiling_data_list=None, application_profiles=None, applications_fixed=None,
                 priority_combinations=None, frequency_combinations=[[0]*5,[2]*5,[6]*5,[11]*5], num_cores_list=[1,3,5]):
    """
    Single-Agent Model-Free RL training (SAMFRL) - FIXED VERSION.

    Only uses Profiler agent for frequency + core allocation.
    Fixed priority (10,10,10) and no thermal/priority agents.

    Returns:
        (ts, makespan_per_app, energy_per_app, temperature_per_app)
    """

    logging.info("=" * 80)
    logging.info("Starting SAMFRL Training (Single-Agent Model-Free) - FIXED VERSION")
    logging.info("=" * 80)

    if profiling_data_list is None:
        profiling_data_list = []
    if applications_fixed is None:
        applications_fixed = []

    # Fix the priority combination to a single constant (10,10,10)
    priority_combinations = [(10,10,10)]

    if len(applications_fixed) == 0:
        logging.error("No applications.")
        return None

    # ---------------- SINGLE AGENT (Profiler) ----------------
    profiler_state = get_profiler_agent_state(profiling_data_list)
    if profiler_state is None:
        logging.error("Could not parse initial profiler state.")
        return None

    profiler_actions = []
    for freq_combo in frequency_combinations:
        for nc in num_cores_list:
            profiler_actions.append((freq_combo[0], nc))
    profiler_action_space = len(profiler_actions)

    if not profiling_data_list:
        logging.error("No initial profiling data available.")
        return None

    num_apps = len(applications_fixed)
    fixed_priority_tuple = tuple([10]*num_apps)
    priority_tuple_int = int(''.join(str(x) for x in fixed_priority_tuple))

    # Create the single ReplayBuffer for profiler
    profiler_memory = ReplayBuffer(mem_size, len(profiler_state), profiler_action_space)
    # Create the single DQNAgent for profiler
    profiler_agent = DQNAgent(state_size=len(profiler_state), action_size=profiler_action_space,
                              load_model=load_model, discount_factor=discount_factor, epsilon=epsilon,
                              epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, epsilon_start=epsilon_start,
                              epsilon_end=epsilon_end, batch_size=batch_size, memsize=mem_size, learning_rate=learning_rate,
                              memory=profiler_memory)

    # Optionally load a pre-trained model and replay memory (profiler only)
    if load_model:
        profiler_memory.load_data(os.path.join(save_dir, "samfrl_profiler_state_transitions.csv.npz"))
        profiler_agent_model_path = os.path.join(save_dir, "samfrl_profiler_model_data.weights.h5")

        if os.path.exists(profiler_agent_model_path):
            profiler_agent.model.load_weights(profiler_agent_model_path)

        logging.info(f"Loaded real state transitions {profiler_memory.mem_cntr}")

    # Tracking data structures
    ts = []
    makespan_per_app = [[] for _ in applications_fixed]
    energy_per_app = [[] for _ in applications_fixed]
    temperature_per_app = [[] for _ in applications_fixed]
    branch_misses_per_app = [[] for _ in applications_fixed]
    cache_misses_per_app = [[] for _ in applications_fixed]
    freq_per_app = [[] for _ in applications_fixed]
    cores_per_app = [[] for _ in applications_fixed]

    total_makespan_data = []
    total_energy_data = []
    total_branch_misses_data = []
    total_cache_misses_data = []
    profiler_qmax_data = []
    profiler_loss_data = []

    profiler_rewards = []
    total_rewards = []
    priority_data = []
    thermal_data = []
    num_cores_data = []
    cores_combination_data = []  # Bitmask for plotting
    cores_str_data = []  # String representation for CSV
    freq_data = []
    priority_values_data = []

    reward_sum = 0.0
    count_steps = 0
    avg_temperature_prev = target_temp

    CoreList = [1,2,3,4,5]
    collected_data = []

    # Use server_name_main if provided (from server_combined.py), otherwise generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if server_name_main and server_name_main != 'samfrl_main.csv':
        csv_filename = server_name_main if os.path.isabs(server_name_main) else os.path.join(save_dir, server_name_main)
    else:
        csv_filename = os.path.join(save_dir, f"SAMFRL_{timestamp}_{experiment_time}ep_beta{beta}_lr{learning_rate}_eps{epsilon_min}_batch{batch_size}.csv")

    iterations_since_last_save = 0
    checkpoint_interval = save_repetition

    # =========================================================================
    # Initialize Online Plotting using unified OnlinePlotter
    # =========================================================================
    plotter = OnlinePlotter(
        data_keys=DATA_KEYS,
        experiment_time=experiment_time,
        module_name="SAMFRL",
        save_dir=save_dir,
        server_name_1=server_name_1,
        server_name_2=server_name_2,
        baseline_label="SAMBRL (Model-Based)",
        previous_label="SAMFRL (Previous)"
    )

    # =========================================================================
    # TRAINING LOOP - Single-Agent Model-Free
    # =========================================================================
    logging.info("Starting main training loop (SAMFRL - Single-Agent Model-Free)...")

    for t in range(experiment_time):
        logging.info(f"\n{'='*60}")
        logging.info(f"SAMFRL Iteration {t + 1}/{experiment_time}")
        logging.info(f"{'='*60}")

        # Get action from profiler agent
        profiler_action_id = profiler_agent.get_action(profiler_state)
        freq_step, num_cores = profiler_actions[profiler_action_id]

        chosen_freq_comb = [freq_step] * 5
        available_cores = CoreList[:num_cores]
        num_cores_data.append(num_cores)

        # Build application instances
        applications_run = []
        for app_idx, app in enumerate(applications_fixed):
            cores_assigned = available_cores.copy()
            cores_assigned_str = ','.join(map(str, cores_assigned))
            frequencies_assigned = chosen_freq_comb[:len(cores_assigned)]
            app_instance = {
                'id': app['id'],
                'benchmark': app.get('benchmark', app.get('path', '')),
                'variant': app.get('variant', ''),
                'input_arg': app.get('input_arg', '5'),
                'app_args': app.get('app_args', app.get('input_arg', '5')),
                'priority': fixed_priority_tuple[app_idx],
                'frequencies': frequencies_assigned,
                'num_cores': len(cores_assigned),
                'cores': cores_assigned_str,
                'action': 'run'
            }
            applications_run.append(app_instance)

        logging.info(f"Actions: freq={freq_step}, cores={num_cores}, priority={fixed_priority_tuple}, replay_buffer={profiler_memory.mem_cntr}")

        # Send to client
        send_msg_dict = {
            'applications': applications_run,
            'run_mode': 'parallel'
        }
        send_msg = json.dumps(send_msg_dict)
        client_socket.send((send_msg + '\n').encode())
        logging.info("Sent apps to client.")

        # Receive from client
        data_received = False
        recv_buffer = ''
        start_time_wait = time.time()
        timeout = 600
        new_profiling_data_list = []
        while not data_received and time.time() - start_time_wait < timeout:
            data = client_socket.recv(4096)
            if data:
                recv_buffer += data.decode()
                while '\n' in recv_buffer:
                    msg, recv_buffer = recv_buffer.split('\n', 1)
                    try:
                        msg_json = json.loads(msg)
                    except:
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

        if not data_received:
            logging.error("Timeout waiting for data.")
            continue

        profiling_data_list = new_profiling_data_list

        # Normalize profiling data to compute avg_temp_after from thermal zones
        profiling_data_list = [normalize_profiling_data(d) for d in profiling_data_list]

        # Compute union of core combinations (both bitmask and string)
        union_cores = 0
        all_cores_set = set()
        for d in profiling_data_list:
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

        # Aggregate results
        makespans = [d.get('time_elapsed',0.0) for d in profiling_data_list]
        total_makespan = sum(makespans)
        energies = [d.get('total_energy_consumption',0.0) for d in profiling_data_list]
        total_energy = sum(energies)

        # Get temperature with fallbacks
        temperatures = [get_temperature_with_fallbacks(d, target_temp) for d in profiling_data_list]
        avg_temperature = float(np.mean(temperatures)) if temperatures else float(target_temp)
        thermal_data.append(avg_temperature)

        target_makespans = [d.get('makespan_all_cores_frequency_11', 0.0) for d in profiling_data_list if d.get('makespan_all_cores_frequency_11') is not None]
        total_target_makespan = sum(target_makespans) if target_makespans else total_makespan
        target_energies = [d.get('energy_all_cores_frequency_0', 0.0) for d in profiling_data_list if d.get('energy_all_cores_frequency_0') is not None]
        total_target_energy = sum(target_energies) if target_energies else total_energy

        # Track per-app metrics
        for app_idx, d in enumerate(profiling_data_list):
            mk = d.get('time_elapsed',0.0)
            makespan_per_app[app_idx].append(mk)
            en = d.get('total_energy_consumption',0.0)
            energy_per_app[app_idx].append(en)
            temp = get_temperature_with_fallbacks(d, target_temp)
            temperature_per_app[app_idx].append(temp)
            bmiss = d.get('branch_misses',0) or 0  # Handle None
            cmiss = d.get('cache_misses',0) or 0  # Handle None
            branch_misses_per_app[app_idx].append(bmiss)
            cache_misses_per_app[app_idx].append(cmiss)
            freq_per_app[app_idx].append(freq_step)
            app_inst = applications_run[app_idx]
            cores_count = app_inst['num_cores']
            cores_per_app[app_idx].append(cores_count)

        total_makespan_data.append(total_makespan)
        total_energy_data.append(total_energy)
        # Handle None values for branch/cache misses
        total_branch_misses = sum(d.get('branch_misses',0) or 0 for d in profiling_data_list)
        total_cache_misses = sum(d.get('cache_misses',0) or 0 for d in profiling_data_list)
        total_branch_misses_data.append(total_branch_misses)
        total_cache_misses_data.append(total_cache_misses)

        # Compute reward
        profiler_reward = get_reward_profiler(
            makespan=total_makespan,
            avg_energy_consumption=total_energy,
            target_makespan=total_target_makespan,
            target_energy=total_target_energy,
            beta=beta
        )

        total_reward = profiler_reward

        reward_sum += total_reward
        count_steps += 1
        total_rewards.append(profiler_reward)
        priority_data.append(priority_tuple_int)
        priority_values_data.append(np.mean(fixed_priority_tuple))

        logging.info(f"[SAMFRL iter={t}] Temp={avg_temperature:.1f}C, BranchMiss={total_branch_misses}, CacheMiss={total_cache_misses}")

        makespans_new = [pd_.get('time_elapsed',0.0) for pd_ in profiling_data_list]
        total_mk_new = sum(makespans_new)
        avg_temperature_prev = avg_temperature

        # New state for profiler
        new_profiler_state = get_profiler_agent_state(profiling_data_list)
        if new_profiler_state is None:
            new_profiler_state = profiler_state.copy()

        # Store transition in single-agent replay buffer
        profiler_memory.store_observations(profiler_state, profiler_action_id, profiler_reward, new_profiler_state)

        # Sample from memory & train if we have enough data
        if profiler_memory.mem_cntr >= batch_size:
            mini_batch = profiler_memory.sample_observation(batch_size)
            if mini_batch is not None:
                profiler_agent.train_model(mini_batch)
                # Periodically update the target network
                if t % save_repetition == 0:
                    profiler_agent.update_target_model()

        # Q-max for profiler
        p_qval = profiler_agent.model.predict(
            profiler_memory.max_normalize(np.array([profiler_state], dtype=np.float32), mode='state')
            if profiler_memory.initialized else np.array([profiler_state],dtype=np.float32),
            verbose=0
        )[0]
        profiler_qmax = np.amax(p_qval)
        profiler_qmax_data.append(profiler_qmax)
        profiler_loss_data.append(profiler_agent.currentLoss)

        ts.append(t)

        # Collect data for CSV (one row per iteration)
        time_stamp = time.time()
        row_data = {
            'iteration': t,
            'run_mode': 'parallel',
            'num_cores': num_cores,
            'priority_tuple': priority_tuple_int,
            'makespan': total_makespan,
            'energy': total_energy,
            'branchmisses': total_branch_misses,
            'cachemisses': total_cache_misses,
            'cores': cores_str_data[-1],  # Use string representation for CSV readability
            'freq': freq_step,
            'qmax': profiler_qmax,
            'loss': profiler_agent.currentLoss,
            'reward': total_reward,
            'thermal': avg_temperature,
            'priority_combination': priority_tuple_int,
            'priority': np.mean(fixed_priority_tuple),
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

        # =====================================================================
        # Online Plotting - Update using unified OnlinePlotter
        # =====================================================================
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
            total_rewards=total_rewards
        )

        # Update plots using unified OnlinePlotter
        plotter.update(current_run_data_map, t)

        # Save models and data periodically
        iterations_since_last_save += 1
        if iterations_since_last_save >= checkpoint_interval and checkpoint_interval != 0:
            if save_model:
                os.makedirs(save_dir, exist_ok=True)
                # Save only the profiler model & memory
                profiler_agent.model.save_weights(os.path.join(save_dir, "samfrl_profiler_model_data.weights.h5"))
                profiler_memory.save_data(os.path.join(save_dir, "samfrl_profiler_state_transitions.csv"))
                logging.info("[Save profiler model and replay memory]")

            if collected_data:
                csv_path = os.path.join(save_dir, server_name_main)
                df_csv = pd.DataFrame(collected_data)
                df_csv.to_csv(csv_path, index=False)
                logging.info(f"Saved collected data to '{csv_path}'.")

            iterations_since_last_save = 0

        profiler_state = new_profiler_state

    # =========================================================================
    # Final Save
    # =========================================================================
    if collected_data:
        csv_path = os.path.join(save_dir, csv_filename)
        df_csv = pd.DataFrame(collected_data)
        df_csv.to_csv(csv_path, index=False)
        logging.info(f"Saved collected data to '{csv_path}'.")

    # Save final plot using unified plotter - derive from csv_filename for consistency
    plot_filename = os.path.basename(csv_filename).replace('.csv', '_plot.png')
    plotter.save(os.path.join(save_dir, plot_filename))
    plotter.close()  # Free memory

    logging.info("SAMFRL training completed.")
    logging.info("=" * 80)
    logging.info("SAMFRL Training Complete (Single-Agent Model-Free)")
    logging.info("=" * 80)

    # Return values for combined_tx2_fixed.py
    return ts, makespan_per_app, energy_per_app, temperature_per_app


if __name__ == '__main__':
    logging.info("SAMFRL (Single-Agent Model-Free) script loaded")
    logging.info("This script should be called from combined_tx2_fixed.py")
