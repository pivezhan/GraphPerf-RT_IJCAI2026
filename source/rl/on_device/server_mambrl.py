#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAMBRL with D3QN - Model-Based Multi-Agent RL for Jetson TX2
FIXED VERSION - Compatible with client_evaluate_tx2.py

KEY FIXES FROM mambrl_d3qn_tx2.py:
=================================
1. app_instance format now includes 'app_args' field that client expects
2. Communication matches server_mambrl_d3qn.py + client_evaluate_tx2.py pattern
3. All original training features preserved (adaptive rewards, exploration boost, etc.)

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

TRAINING FEATURES (unchanged):
==============================
1. Adaptive reward system with improvement bonuses
2. Best performance tracking and curriculum learning
3. Higher epsilon minimum (0.10) for continued exploration
4. Exploration boost when stuck
5. Progress bonuses and dynamic target adaptation
6. Model-based planning with environment model
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

# Import unified plotting module
from live_plotter import (
    OnlinePlotter, create_data_map,
    load_historical_data, extract_server_name, parse_hyperparams_from_filename
)
from server_combined import DATA_KEYS
import socket
import threading
from copy import deepcopy
from scipy import stats

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory where this script is located for save paths
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "save_model")

STATE_SIZE = None

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
            delta_key = f'thermal_zone{i}_delta'
            if delta_key in profiling_data:
                delta_val = profiling_data.get(delta_key, 0.0)
                temp_deltas.append(float(delta_val) if delta_val is not None else 0.0)
            else:
                temp_deltas.append(0.0)

        # Build state vector
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
            *temps_after[:5],
            *temp_deltas,
            avg_temp_delta
        ], dtype=np.float32)

        if STATE_SIZE is None:
            STATE_SIZE = len(state)
            logging.info(f"Determined state size: {STATE_SIZE}")

        return state, total_energy_consumption, avg_temp_after, time_elapsed, branch_misses, cache_misses, target_makespan, target_energy, parallelism_level
    except Exception as e:
        logging.error(f"Error parsing state: {e}", exc_info=True)
        return None


def priority_tuple_to_int(tup):
    """Convert priority tuple (e.g., (50,50,50)) to an integer label."""
    return int(''.join(str(x) for x in tup))


# =============================================================================
# REVISED Reward Functions with Adaptive Learning
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
    Temperature agent reward (Fig. 1(b)-style, constraint-based, linear):

    - If avg_temperature <= target_temp:
        r_temp = 0       (no penalty, no bonus)
    - If avg_temperature >  target_temp:
        r_temp = -k * (avg_temperature - target_temp)

    where k = overheat_slope.

    Optionally, an extra penalty can be added when crossing from below
    to above the threshold (cross_penalty), but default is 0.0.

    This matches the paper's description: only failures to keep the
    temperature under the threshold are penalized; otherwise there is
    no penalty.
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
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
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
# Environment Model (for Model-Based RL)
# =============================================================================
class FCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class EnvironmentModel(nn.Module):
    def __init__(self, input_dim, output_dim, batch_size, num_apps, thermal_action_space, priority_action_space):
        super().__init__()
        learning_rate = 0.001
        self.model = FCNModel(input_dim, 128, output_dim)
        self.state_size = output_dim
        self.num_apps = num_apps
        self.thermal_action_space = thermal_action_space
        self.priority_action_space = priority_action_space

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size

    def train_model(self, replay_buffer, epochs=100):
        for epoch in range(epochs):
            all_observations = replay_buffer.sample_observation(
                min(replay_buffer.mem_cntr, self.batch_size), random=False
            )
            if all_observations is None:
                return

            states = np.array([obs[0] for obs in all_observations], dtype=np.float32)
            actions = np.array([obs[1] for obs in all_observations], dtype=np.float32)
            next_states = np.array([obs[3] for obs in all_observations], dtype=np.float32)

            normal_states = replay_buffer.max_normalize(states, mode='state')
            normal_actions = replay_buffer.max_normalize(actions, mode='action')
            normal_next_states = replay_buffer.max_normalize(next_states, mode='state')

            normal_states = T.tensor(normal_states, dtype=T.float32)
            normal_actions = T.tensor(normal_actions, dtype=T.float32).unsqueeze(1)
            normal_next_states = T.tensor(normal_next_states, dtype=T.float32)

            inputs = T.cat([normal_states, normal_actions], dim=1)
            dataset = TensorDataset(inputs, normal_next_states)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            epoch_loss = 0.0
            for inp, targets in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(inp)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if epoch % 20 == 0:
                logging.info(f'EnvModel Epoch {epoch + 1}/{epochs}: Average Loss = {avg_loss:.4f}')

    def predict_next_state(self, replay_buffer, state, action):
        """Predict next state given current state and scalar action."""
        normal_state = replay_buffer.max_normalize(state, mode='state')
        normal_action = replay_buffer.max_normalize([action], mode='action')
        normal_state_tensor = T.tensor(normal_state, dtype=T.float32).unsqueeze(0)
        normal_action_tensor = T.tensor(normal_action, dtype=T.float32).unsqueeze(0)
        normal_input_tensor = T.cat([normal_state_tensor, normal_action_tensor], dim=1)

        with T.no_grad():
            normal_next_state_tensor = self.model(normal_input_tensor)

        denormal_next_state = replay_buffer.max_denormalize(
            normal_next_state_tensor.detach().cpu().numpy(), mode='state'
        )[0]
        return denormal_next_state

    def long_horizon_reward_estimation(self, replay_buffer, gen_buffer, plan_count,
                                       target_makespan, target_energy, target_temp,
                                       beta=1.0, thermal_weight=0.2, best_makespan=None):
        """
        Generate synthetic trajectories using the learned environment model.

        Uses the SAME global target_makespan and target_energy as the online
        reward for consistency between real and model-generated experience.
        """
        gen_buffer['profiler'].mem_cntr = 0
        gen_buffer['thermal'].mem_cntr = 0
        gen_buffer['priority'].mem_cntr = 0

        samples = replay_buffer.sample_observation(batch_size=50, random=True)
        if samples is None:
            return

        for obs in samples:
            state, action, _, _ = obs
            current_state = state.copy()
            current_action = action

            avg_profiler_reward = 0.0
            avg_thermal_reward = 0.0
            avg_total_reward = 0.0

            avg_temperature_prev = target_temp
            current_makespan = current_state[16] if len(current_state) > 16 else 1.0

            if len(current_state) > 21:
                avg_energy_consumption = float(np.mean(current_state[17:22]))
            else:
                avg_energy_consumption = 1.0

            if len(current_state) > 26:
                avg_temp_after_vals = current_state[22:27]
                avg_temperature = float(np.mean(avg_temp_after_vals))
            else:
                avg_temp_after_vals = np.array([target_temp] * 5, dtype=np.float32)
                avg_temperature = float(target_temp)

            profiler_r, _ = get_reward_profiler(
                current_makespan,
                avg_energy_consumption,
                target_makespan,
                target_energy,
                beta=beta,
                best_makespan=best_makespan
            )
            thermal_r = get_reward_thermal(avg_temperature, avg_temperature_prev, target_temp)
            total_r = profiler_r + thermal_weight * thermal_r

            avg_profiler_reward += profiler_r
            avg_thermal_reward += thermal_r
            avg_total_reward += total_r

            next_state = current_state

            for _ in range(plan_count - 1):
                next_state = self.predict_next_state(replay_buffer, next_state, current_action)

                future_makespan = next_state[16] if len(next_state) > 16 else current_makespan
                future_energy = np.mean(next_state[17:22]) if len(next_state) > 21 else avg_energy_consumption
                future_temp_after = next_state[22:27] if len(next_state) > 26 else avg_temp_after_vals
                future_avg_temp = float(np.mean(future_temp_after))

                p_r, _ = get_reward_profiler(
                    future_makespan,
                    future_energy,
                    target_makespan,
                    target_energy,
                    beta=beta,
                    best_makespan=best_makespan
                )
                th_r = get_reward_thermal(future_avg_temp, avg_temperature, target_temp)
                tot_r = p_r + thermal_weight * th_r

                avg_profiler_reward += p_r
                avg_thermal_reward += th_r
                avg_total_reward += tot_r

                avg_temperature = future_avg_temp

            avg_profiler_reward /= float(plan_count)
            avg_thermal_reward /= float(plan_count)
            avg_total_reward /= float(plan_count)

            gen_buffer['profiler'].store_observations(state, action, avg_profiler_reward, next_state)

            thermal_action = random.randint(0, self.thermal_action_space - 1)
            if len(state) > 32:
                thermal_s = np.array(
                    [avg_temperature, np.mean(state[27:32]), avg_temperature, np.mean(state[27:32])],
                    dtype=np.float32
                )
            else:
                thermal_s = np.array([target_temp, 0.0, target_temp, 0.0], dtype=np.float32)

            if len(next_state) > 32:
                thermal_s_ = np.array(
                    [avg_temperature, np.mean(next_state[27:32]), avg_temperature, np.mean(next_state[27:32])],
                    dtype=np.float32
                )
            else:
                thermal_s_ = thermal_s.copy()

            gen_buffer['thermal'].store_observations(thermal_s, thermal_action, avg_thermal_reward, thermal_s_)

            # Priority agent uses combined total reward (profiler + thermal_weight * thermal)
            priority_action = random.randint(0, self.priority_action_space - 1)

            total_makespan = current_makespan
            future_total_makespan = next_state[16] if len(next_state) > 16 else current_makespan

            current_app_makespans = [current_makespan / max(self.num_apps, 1)] * self.num_apps
            future_app_makespans = [future_total_makespan / max(self.num_apps, 1)] * self.num_apps

            priority_s = np.array([total_makespan] + current_app_makespans, dtype=np.float32)
            priority_s_ = np.array([future_total_makespan] + future_app_makespans, dtype=np.float32)

            gen_buffer['priority'].store_observations(priority_s, priority_action, avg_total_reward, priority_s_)


# =============================================================================
# Helper Functions for Agent States
# =============================================================================
def get_priority_agent_state(app_makespans, total_makespan):
    """Create state for priority agent."""
    return np.array([total_makespan] + app_makespans, dtype=np.float32)


def get_thermal_agent_state(applications_data):
    """Create state for thermal agent from per-application avg temperature and delta."""
    if len(applications_data) == 0:
        return np.zeros(4, dtype=np.float32)

    avg_temp_after_vals = []
    avg_temp_delta_vals = []

    for ad in applications_data:
        avg_temp_after_vals.append(ad.get('avg_temp_after', 0.0))
        avg_temp_delta_vals.append(ad.get('avg_temp_delta', 0.0))

    avg_temp = float(np.mean(avg_temp_after_vals)) if avg_temp_after_vals else 0.0
    avg_delta = float(np.mean(avg_temp_delta_vals)) if avg_temp_delta_vals else 0.0

    return np.array([avg_temp, avg_delta, avg_temp, avg_delta], dtype=np.float32)


def get_profiler_agent_state(applications_data):
    """Create state for profiler agent using the first application's detailed metrics."""
    if not applications_data:
        return None
    parsed = parse_state(applications_data[0])
    if parsed is None:
        return None
    state, _, _, _, _, _, _, _, _ = parsed
    return state


# =============================================================================
# Statistics Calculation Functions
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


def export_statistics_to_csv(stats_data, filename):
    """Export comprehensive statistics to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std Dev', '95% CI Lower', '95% CI Upper', 'Min', 'Max'])

        for metric_name, values in stats_data.items():
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                ci_mean, ci_lower, ci_upper = calculate_confidence_interval(values)
                min_val = np.min(values)
                max_val = np.max(values)

                writer.writerow([
                    metric_name,
                    f'{mean:.4f}',
                    f'{std:.4f}',
                    f'{ci_lower:.4f}',
                    f'{ci_upper:.4f}',
                    f'{min_val:.4f}',
                    f'{max_val:.4f}'
                ])

    logging.info(f"Statistics exported to {filename}")


# =============================================================================
# Misc Utility
# =============================================================================
def load_historical_data(filepath, data_keys, target_rows=None):
    """Load historical training data from CSV for comparison plotting.

    Args:
        filepath: Path to the CSV file
        data_keys: List of column names to load
        target_rows: If specified and data has more rows, aggregate to this count
                    (for cross-module comparison where MAMFRL has different row count)

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
            if target_rows and actual_rows != target_rows:
                if actual_rows > target_rows and actual_rows % target_rows == 0:
                    # Aggregate by taking mean of each group
                    group_size = actual_rows // target_rows
                    logging.info(f"Aggregating {actual_rows} rows to {target_rows} (group_size={group_size})")

                    for key in data_keys:
                        if key in df.columns:
                            values = df[key].tolist()
                            aggregated = []
                            for i in range(0, len(values), group_size):
                                group = values[i:i+group_size]
                                aggregated.append(np.mean(group))
                            data[key] = aggregated
                elif actual_rows < target_rows:
                    # MAMFRL has fewer rows - use directly (MAMBRL will have more data points)
                    logging.info(f"Using {actual_rows} rows directly (fewer than target {target_rows})")
                    for key in data_keys:
                        if key in df.columns:
                            data[key] = df[key].tolist()
                else:
                    logging.warning(f"Cannot aggregate {actual_rows} rows to {target_rows} (not evenly divisible)")
                    for key in data_keys:
                        if key in df.columns:
                            data[key] = df[key].tolist()
            else:
                # No aggregation needed - load directly
                for key in data_keys:
                    if key in df.columns:
                        data[key] = df[key].tolist()

        except Exception as e:
            logging.warning(f"Failed to load historical data: {e}")
    else:
        logging.info(f"No historical data found at {filepath}")
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


def pretty(obj):
    """Pretty print JSON objects."""
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


# =============================================================================
# Main Training Function (FULLY REVISED)
# =============================================================================
def train_fixed_mambrl_d3qn(client_socket, data_keys, experiment_time=100, clock_change_time=30, beta=1.0,
                     load_model=False, learn_count=32, plan_count=100, mem_size=100000, learning_rate=0.1,
                     discount_factor=0.99, epsilon=1.0, epsilon_decay=0.90, epsilon_min=0.10, epsilon_start=1.0,
                     epsilon_end=0.01, reset_learning_rate_value=20, save_repetition=50, save_model=True,
                     batch_size=64, agent_train_start=32, target_temp=50.0, thermal_weight=0.2,
                     server_name_1='', server_name_2='', server_name_main='marbrl_main.csv',
                     profiling_data_list=None, application_profiles=None, applications_fixed=None,
                     priority_combinations=None, frequency_combinations=None, num_cores_list=None,
                     model_train_start=32, real_synthetic_ratio=0.5):
    """
    FULLY REVISED: Train MAMBRL with D3QN agents with adaptive rewards and exploration.

    MODEL-BASED RL PARAMETERS:
    - plan_count: Number of synthetic samples generated per planning phase (default: 100)
    - model_train_start: Real data threshold before environment model training (default: 32)
    - agent_train_start: Total data threshold before agent training starts (default: 32)
    - real_synthetic_ratio: Ratio of real to synthetic data in training batch (default: 0.5)

    NEW FEATURES:
    - Adaptive reward system with improvement bonuses
    - Best performance tracking
    - Exploration boost when stuck
    - Higher epsilon minimum (0.10)
    - All original functionality preserved
    """
    # Use model_train_start if not None, else fall back to learn_count
    if model_train_start is None:
        model_train_start = learn_count
    logging.info("=" * 80)
    logging.info("FULLY REVISED MAMBRL WITH D3QN - ADAPTIVE REWARDS + EXPLORATION FIXES")
    logging.info(f"Beta (makespan-first)         = {beta}")
    logging.info(f"Thermal reward weight (λ_th)  = {thermal_weight}")
    logging.info(f"Epsilon minimum (REVISED)     = {max(epsilon_min, 0.10)}")
    logging.info("-" * 40)
    logging.info("MODEL-BASED RL PARAMETERS:")
    logging.info(f"  Plan count (synthetic data)  : {plan_count}")
    logging.info(f"  Model train start (real data): {model_train_start}")
    logging.info(f"  Agent train start (total)    : {agent_train_start}")
    logging.info(f"  Real/Synthetic ratio         : {real_synthetic_ratio}")
    logging.info("-" * 40)
    logging.info(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
    logging.info("=" * 80)

    if profiling_data_list is None:
        profiling_data_list = []
    if applications_fixed is None:
        applications_fixed = []
    if priority_combinations is None:
        priority_combinations = [(50, 50, 50), (90, 90, 90)]
    if frequency_combinations is None:
        frequency_combinations = [[0] * 5, [2] * 5, [6] * 5, [11] * 5]
    if num_cores_list is None:
        num_cores_list = [1, 3, 5]

    if len(applications_fixed) == 0:
        logging.error("No applications to train on!")
        return

    num_apps = len(applications_fixed)
    logging.info(f"Training with {num_apps} applications")

    # NEW: Initialize best performance tracker
    perf_tracker = BestPerformanceTracker()

    # Agent state sizes
    priority_state_size = num_apps + 1
    thermal_state_size = 4

    # Profiler actions: (freq_level, num_cores)
    profiler_actions = []
    for freq_combo in frequency_combinations:
        for nc in num_cores_list:
            profiler_actions.append((freq_combo[0], nc))
    profiler_action_space = len(profiler_actions)
    logging.info(f"Profiler action space size: {profiler_action_space}")

    # Thermal actions: (core_id, priority_change)
    thermal_actions = []
    for core_id in range(1, 6):
        for pc in [-1, 1]:
            thermal_actions.append((core_id, pc))
    thermal_action_space = len(thermal_actions)
    logging.info(f"Thermal action space size: {thermal_action_space}")

    # Priority actions
    priority_action_space = len(priority_combinations)
    logging.info(f"Priority action space size: {priority_action_space}")

    if not profiling_data_list:
        logging.error("No initial profiling data available!")
        return

    # Initial agent states
    profiler_state = get_profiler_agent_state(profiling_data_list)
    if profiler_state is None:
        logging.error("Could not parse initial profiler state!")
        return

    makespans_init = [pd_.get('time_elapsed', 0.0) for pd_ in profiling_data_list]
    total_makespan_init = sum(makespans_init) if makespans_init else 0.0
    priority_state = get_priority_agent_state(makespans_init, total_makespan_init)
    thermal_state = get_thermal_agent_state(profiling_data_list)

    logging.info(f"Initial profiler state size:  {len(profiler_state)}")
    logging.info(f"Initial priority state size:  {len(priority_state)}")
    logging.info(f"Initial thermal state size:   {len(thermal_state)}")

    # Initial thermal check
    initial_temps = [pd_.get('avg_temp_after', target_temp) for pd_ in profiling_data_list]
    logging.info(f"Initial temperatures: {[f'{t:.2f}' for t in initial_temps]}")
    if all(t == target_temp for t in initial_temps):
        logging.warning("All initial avg_temp_after equal to target_temp – check client thermal data if unexpected.")

    # -------------------------------------------------------------------------
    # GLOBAL TARGETS FROM CLIENT PROFILING (MIN MAKESPAN & ENERGY BASELINE)
    # -------------------------------------------------------------------------
    global_target_makespan = 0.0
    global_target_energy = 0.0

    for pd_ in profiling_data_list:
        mk = pd_.get('makespan_all_cores_frequency_11')
        if mk is not None and mk > 0:
            global_target_makespan += mk
        else:
            logging.warning(
                f"Missing makespan_all_cores_frequency_11 for app {pd_.get('application_id')}, "
                f"will rely on fallback if needed."
            )

        en = pd_.get('energy_all_cores_frequency_0')
        if en is not None and en > 0:
            global_target_energy += en
        else:
            logging.warning(
                f"Missing energy_all_cores_frequency_0 for app {pd_.get('application_id')}, "
                f"will rely on fallback if needed."
            )

    if global_target_makespan <= 0:
        fallback_mk = sum(pd_.get('time_elapsed', 0.0) for pd_ in profiling_data_list)
        global_target_makespan = max(fallback_mk, 1e-3)
        logging.warning(
            "No valid minimum-makespan targets from client; "
            f"using sum of current makespans as target: {global_target_makespan:.3f}s"
        )
    else:
        logging.info(
            f"GLOBAL MINIMUM MAKESPAN target (sum of makespan_all_cores_frequency_11): "
            f"{global_target_makespan:.3f}s"
        )

    if global_target_energy <= 0:
        fallback_energy = sum(pd_.get('total_energy_consumption', 0.0) for pd_ in profiling_data_list)
        global_target_energy = max(fallback_energy, 1e-6)
        logging.warning(
            "No valid energy baseline from client; "
            f"using sum of current energies as target: {global_target_energy:.3f}J"
        )
    else:
        logging.info(
            "GLOBAL ENERGY BASELINE target (sum of energy_all_cores_frequency_0): "
            f"{global_target_energy:.3f}J"
        )

    # -------------------------------------------------------------------------
    # Replay Buffers
    # -------------------------------------------------------------------------
    profiler_memory = ReplayBuffer(mem_size, len(profiler_state), profiler_action_space)
    thermal_memory = ReplayBuffer(mem_size, thermal_state_size, thermal_action_space)
    priority_memory = ReplayBuffer(mem_size, priority_state_size, priority_action_space)

    profiler_gen_memory = ReplayBuffer(mem_size, len(profiler_state), profiler_action_space)
    thermal_gen_memory = ReplayBuffer(mem_size, thermal_state_size, thermal_action_space)
    priority_gen_memory = ReplayBuffer(mem_size, priority_state_size, priority_action_space)

    # -------------------------------------------------------------------------
    # REVISED: Agents with higher epsilon minimum
    # -------------------------------------------------------------------------
    logging.info("Initializing D3QN agents with REVISED exploration parameters...")
    
    # CRITICAL: Ensure epsilon_min is at least 0.10
    epsilon_min_revised = max(epsilon_min, 0.10)
    
    profiler_agent = D3QNAgent(
        state_size=len(profiler_state),
        action_size=profiler_action_space,
        load_model=load_model,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min_revised,  # REVISED
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        batch_size=batch_size,
        memsize=mem_size,
        learning_rate=learning_rate
    )

    thermal_agent = D3QNAgent(
        state_size=thermal_state_size,
        action_size=thermal_action_space,
        load_model=load_model,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min_revised,  # REVISED
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        batch_size=batch_size,
        memsize=mem_size,
        learning_rate=learning_rate
    )

    priority_agent = D3QNAgent(
        state_size=priority_state_size,
        action_size=priority_action_space,
        load_model=load_model,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min_revised,  # REVISED
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        batch_size=batch_size,
        memsize=mem_size,
        learning_rate=learning_rate
    )

    logging.info(f"Agents initialized with epsilon_min={epsilon_min_revised:.3f}")

    # Environment Model
    input_dim = len(profiler_state) + 1
    output_dim = len(profiler_state)
    env_model = EnvironmentModel(input_dim, output_dim, batch_size, num_apps, thermal_action_space, priority_action_space)

    # Load models / memories if requested
    if load_model:
        model_dir = save_dir
        profiler_memory.load_data(os.path.join(model_dir, 'profiler_state_transitions.csv.npz'))
        thermal_memory.load_data(os.path.join(model_dir, 'thermal_state_transitions.csv.npz'))
        priority_memory.load_data(os.path.join(model_dir, 'priority_state_transitions.csv.npz'))

        profiler_gen_memory.load_data(os.path.join(model_dir, 'profiler_gen_state_transitions.csv.npz'))
        thermal_gen_memory.load_data(os.path.join(model_dir, 'thermal_gen_state_transitions.csv.npz'))
        priority_gen_memory.load_data(os.path.join(model_dir, 'priority_gen_state_transitions.csv.npz'))

        if os.path.exists(os.path.join(model_dir, "profiler_model_data.weights.h5")):
            profiler_agent.model.load_weights(os.path.join(model_dir, "profiler_model_data.weights.h5"))
        if os.path.exists(os.path.join(model_dir, "thermal_model_data.weights.h5")):
            thermal_agent.model.load_weights(os.path.join(model_dir, "thermal_model_data.weights.h5"))
        if os.path.exists(os.path.join(model_dir, "priority_model_data.weights.h5")):
            priority_agent.model.load_weights(os.path.join(model_dir, "priority_model_data.weights.h5"))

        logging.info(
            f"Models loaded: profiler={profiler_memory.mem_cntr}, "
            f"thermal={thermal_memory.mem_cntr}, priority={priority_memory.mem_cntr}"
        )

    # -------------------------------------------------------------------------
    # Tracking / Logging Data Structures
    # -------------------------------------------------------------------------
    ts = []
    makespan_per_app = [[] for _ in applications_fixed]
    priority_per_app = [[] for _ in applications_fixed]
    energy_per_app = [[] for _ in applications_fixed]
    temperature_per_app = [[] for _ in applications_fixed]  # Added for comparison with MAMFRL
    branch_misses_per_app = [[] for _ in applications_fixed]
    cache_misses_per_app = [[] for _ in applications_fixed]
    freq_per_app = [[] for _ in applications_fixed]
    cores_per_app = [[] for _ in applications_fixed]

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
    total_rewards = []
    priority_data = []
    thermal_data = []  # Added for proper temperature tracking
    num_cores_data = []  # Added for standardized plotting

    stats_data = {
        'makespan': [],
        'energy': [],
        'temperature': [],
        'profiler_reward': [],
        'thermal_reward': [],
        'total_reward': [],
        'profiler_qmax': [],
        'thermal_qmax': [],
        'priority_qmax': [],
        'profiler_loss': [],
        'thermal_loss': [],
        'priority_loss': [],
        'best_makespan': [],
        'improvements': []
    }

    avg_temperature_prev = target_temp
    collected_data = []

    # Use server_name_main if provided (from server_combined.py), otherwise generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if server_name_main and server_name_main != 'marbrl_main.csv':
        csv_filename = server_name_main if os.path.isabs(server_name_main) else os.path.join(save_dir, server_name_main)
    else:
        csv_filename = os.path.join(save_dir, f"MAMBRL_D3QN_{timestamp}_{experiment_time}ep_beta{beta}_lr{learning_rate}_eps{epsilon_min}_batch{batch_size}.csv")

    iterations_since_last_save = 0
    checkpoint_interval = save_repetition

    CoreList = [1, 2, 3, 4, 5]

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

    # Setup live figure using unified OnlinePlotter (will be initialized after loading historical data)
    plotter = None  # Deferred initialization until after historical data loading

    # =========================================================================
    # Load historical data for comparison - SUPPORT CROSS-MODULE COMPARISON
    # =========================================================================
    # MAMBRL saves one row per application per iteration (e.g., 300 rows for 100 experiments with 3 apps)
    # MAMFRL saves one row per iteration (e.g., 100 rows for 100 experiments)
    # For cross-module comparison (MAMBRL vs MAMFRL), MAMFRL data can be used directly.
    # For same-module comparison, we allow different row counts if exp matches.
    actual_server_name_1 = ""
    actual_server_name_2 = ""
    target_rows_1 = None  # For tracking if aggregation/expansion needed
    target_rows_2 = None

    def get_csv_row_count(filepath):
        """Count actual data rows in CSV (excluding header)."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return sum(1 for _ in f) - 1  # Subtract header
        except Exception:
            pass
        return 0

    def is_cross_module(filepath):
        """Check if the file is from a different module (MAMFRL for MAMBRL comparison)."""
        basename = os.path.basename(filepath).upper()
        return "MAMFRL" in basename  # MAMBRL is loading MAMFRL data

    # Calculate expected MAMBRL row count (rows per iteration * experiment_time)
    # MAMBRL saves 3 rows per iteration (one per app)
    num_apps = len(applications_fixed) if applications_fixed else 3
    expected_mambrl_rows = experiment_time * num_apps

    # server_name_1: Usually cross-module (MAMFRL data for comparison)
    if server_name_1:
        hp1 = parse_hyperparams_from_filename(server_name_1)
        row_count_1 = get_csv_row_count(server_name_1)

        if hp1 and hp1.get('exp') == experiment_time:
            if is_cross_module(server_name_1):
                # Cross-module (MAMFRL) - use directly, it has experiment_time rows
                actual_server_name_1 = server_name_1
                target_rows_1 = None  # No aggregation needed for MAMFRL
                logging.info(f"MAMBRL: Using cross-module data 1 (MAMFRL, exp={experiment_time}, rows={row_count_1}): {os.path.basename(server_name_1)}")
            elif row_count_1 == expected_mambrl_rows:
                # Same module (MAMBRL) with expected row count
                actual_server_name_1 = server_name_1
                target_rows_1 = None  # No aggregation needed
                logging.info(f"MAMBRL: Using same-module data 1 (exp={experiment_time}, rows={row_count_1}): {os.path.basename(server_name_1)}")
            else:
                logging.warning(f"MAMBRL: Skipping historical data 1 (exp={hp1.get('exp')}, rows={row_count_1}, expected_rows={expected_mambrl_rows}): {os.path.basename(server_name_1)}")
        else:
            logging.warning(f"MAMBRL: Skipping historical data 1 (exp mismatch, need {experiment_time}): {os.path.basename(server_name_1)}")

    # server_name_2: Usually same-module (previous MAMBRL data)
    if server_name_2:
        hp2 = parse_hyperparams_from_filename(server_name_2)
        row_count_2 = get_csv_row_count(server_name_2)

        if hp2 and hp2.get('exp') == experiment_time:
            if is_cross_module(server_name_2):
                # Cross-module (MAMFRL) - use directly
                actual_server_name_2 = server_name_2
                target_rows_2 = None
                logging.info(f"MAMBRL: Using cross-module data 2 (MAMFRL, exp={experiment_time}, rows={row_count_2}): {os.path.basename(server_name_2)}")
            elif row_count_2 == expected_mambrl_rows:
                # Same module (MAMBRL) with expected row count
                actual_server_name_2 = server_name_2
                target_rows_2 = None
                logging.info(f"MAMBRL: Using same-module data 2 (exp={experiment_time}, rows={row_count_2}): {os.path.basename(server_name_2)}")
            else:
                logging.warning(f"MAMBRL: Skipping historical data 2 (exp={hp2.get('exp')}, rows={row_count_2}, expected_rows={expected_mambrl_rows}): {os.path.basename(server_name_2)}")
        else:
            logging.warning(f"MAMBRL: Skipping historical data 2 (exp mismatch, need {experiment_time}): {os.path.basename(server_name_2)}")

    # Load historical data
    hist_data_1 = load_historical_data(
        actual_server_name_1, data_keys, target_rows=target_rows_1
    ) if actual_server_name_1 else {}

    hist_data_2 = load_historical_data(
        actual_server_name_2, data_keys, target_rows=target_rows_2
    ) if actual_server_name_2 else {}

    # Update server_name_1/2 to actual values used for legend labels
    server_name_1 = actual_server_name_1
    server_name_2 = actual_server_name_2

    # Initialize unified OnlinePlotter now that historical data is loaded
    # Note: OnlinePlotter will load data internally, but we keep the custom loading above
    # for validation and logging purposes. The plotter handles both cross-module and same-module cases.
    plotter = OnlinePlotter(
        data_keys=data_keys,
        experiment_time=experiment_time,
        module_name="MAMBRL_D3QN",
        save_dir=save_dir,
        server_name_1=os.path.basename(server_name_1) if server_name_1 else "",
        server_name_2=os.path.basename(server_name_2) if server_name_2 else "",
        baseline_label="MAMFRL (Model-Free)" if server_name_1 and "MAMFRL" in server_name_1.upper() else None,
        previous_label="MAMBRL (Previous)" if server_name_2 else None
    )

    gen_buffer = {
        'profiler': profiler_gen_memory,
        'thermal': thermal_gen_memory,
        'priority': priority_gen_memory
    }

    logging.info("Starting main training loop...")
    logging.info(f"Using GLOBAL target_makespan={global_target_makespan:.3f}s, "
                 f"GLOBAL target_energy={global_target_energy:.3f}J")

    for t in range(experiment_time):
        logging.info("\n" + "=" * 60)
        logging.info(f"MAMBRL Iteration {t + 1}/{experiment_time}")
        logging.info(f"⏱  Starting iteration at {time.strftime('%H:%M:%S')}")
        
        # NEW: Check if stuck and boost exploration
        if perf_tracker.needs_exploration_boost(threshold=20):
            logging.warning(f"⚠️  NO IMPROVEMENT for {perf_tracker.iterations_since_improvement} iterations!")
            logging.warning(f"   Boosting exploration: epsilon {profiler_agent.epsilon:.3f} -> {min(profiler_agent.epsilon * 1.5, 0.5):.3f}")
            profiler_agent.epsilon = min(profiler_agent.epsilon * 1.5, 0.5)
            thermal_agent.epsilon = min(thermal_agent.epsilon * 1.5, 0.5)
            priority_agent.epsilon = min(priority_agent.epsilon * 1.5, 0.5)
        
        # Display status
        logging.info(f"Epsilon: {profiler_agent.epsilon:.3f}, Best makespan: {perf_tracker.best_makespan:.3f}s")
        logging.info(f"Iterations since improvement: {perf_tracker.iterations_since_improvement}")
        logging.info("=" * 60)

        # -------------------------------------------------------------
        # Agent actions (with exploration bonus)
        # -------------------------------------------------------------
        profiler_action_id = profiler_agent.get_action(profiler_state, add_exploration_bonus=True)
        freq_step, num_cores = profiler_actions[profiler_action_id]

        thermal_action_id = thermal_agent.get_action(thermal_state, add_exploration_bonus=True)
        core_id_change, priority_change = thermal_actions[thermal_action_id]
        priority_queue.update_core_priority(core_id_change, priority_change)

        priority_action_id = priority_agent.get_action(priority_state, add_exploration_bonus=True)
        current_priority_tuple = priority_combinations[priority_action_id]
        priority_tuple_int = priority_tuple_to_int(current_priority_tuple)

        logging.info(f"Actions: freq={freq_step}, cores={num_cores}, priority={current_priority_tuple}")

        chosen_freq_comb = [freq_step] * 5

        applications_run = []
        available_cores = generate_set_of_cores(num_cores, priority_queue.prioritized_cores_set, CoreList)
        initial_available_cores = available_cores.copy()
        assigned_cores_dict = {}
        cores_assigned_list = []
        level_of_parallelism_list = []

        app_with_priority = [(current_priority_tuple[idx], applications_fixed[idx])
                             for idx in range(len(applications_fixed))]
        app_with_priority_sorted = sorted(app_with_priority, key=lambda x: x[0], reverse=True)

        for priority_val, app in app_with_priority_sorted:
            app_id = app['id']
            # Use str() for consistency with profiling phase key format
            app_input_arg = str(app.get('app_args', app.get('input_arg', '')))
            profile_key = (app_id, app_input_arg)
            app_profile = application_profiles.get(profile_key) if application_profiles else None

            if app_profile is None:
                logging.warning(f"No profiling data for app ID {app_id} with args '{app_input_arg}'. Using default parallelism.")
                level_of_parallelism = 1.0
            else:
                level_of_parallelism = app_profile.get('parallelism_level', 1.0)
                if level_of_parallelism is None:
                    level_of_parallelism = 1.0

            num_cores_to_allocate = max(int(level_of_parallelism), 1)
            num_cores_to_allocate = min(num_cores, num_cores_to_allocate)

            if len(available_cores) >= num_cores_to_allocate:
                cores_assigned = random.sample(available_cores, num_cores_to_allocate)
                for core in cores_assigned:
                    available_cores.remove(core)
            else:
                cores_assigned = available_cores.copy()
                available_cores.clear()
                num_cores_needed = num_cores_to_allocate - len(cores_assigned)
                remaining_cores = [core for core in initial_available_cores if core not in cores_assigned]
                if len(remaining_cores) >= num_cores_needed:
                    additional_cores = random.sample(remaining_cores, num_cores_needed)
                    cores_assigned.extend(additional_cores)
                else:
                    cores_assigned.extend(remaining_cores)

            cores_assigned = list(set(cores_assigned))
            cores_assigned_str = ','.join(map(str, cores_assigned))
            cores_assigned_list.append(cores_assigned_str)
            level_of_parallelism_list.append(level_of_parallelism)

            frequencies_assigned = chosen_freq_comb[:len(cores_assigned)]

            # FIXED: Include 'app_args' that client_evaluate_tx2.py expects
            # Client's application_worker() uses: input_arg = str(app.get("input_arg", ""))
            # and also app_args for compatibility
            app_instance = {
                'id': app['id'],
                'benchmark': app['benchmark'],
                'variant': app['variant'],
                'input_arg': app['input_arg'],
                'app_args': app.get('app_args', app.get('input_arg', '')),  # CRITICAL: client uses this
                'priority': priority_val,
                'frequencies': frequencies_assigned,
                'num_cores': len(cores_assigned),
                'cores': cores_assigned_str,
                'action': 'run'
            }
            applications_run.append(app_instance)
            assigned_cores_dict[app_id] = cores_assigned

        logging.info(
            f"Core allocation: Freq={freq_step}, Cores={num_cores}, "
            f"Assigned={cores_assigned_list}, Parallelism={level_of_parallelism_list}"
        )

        # -------------------------------------------------------------
        # Send to client
        # -------------------------------------------------------------
        send_msg_dict = {
            'applications': applications_run,
            'run_mode': 'parallel'
        }
        send_msg = json.dumps(send_msg_dict)
        client_socket.send((send_msg + '\n').encode())
        logging.info(f"✉️  Sent {len(applications_run)} applications to client at {time.strftime('%H:%M:%S')}")

        # -------------------------------------------------------------
        # Receive from client (pattern from server_mambrl_2.py)
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
                            logging.warning("Received malformed JSON. Skipping.")
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
                            logging.info("Received results from client")
                        else:
                            logging.warning("Profiling data length mismatch. Waiting for complete data.")
                            continue
                else:
                    time.sleep(1)
            except socket.timeout:
                logging.error("Socket timed out while waiting for data.")
                break
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                break

        if not data_received:
            logging.error("Timeout waiting for data.")
            continue

        profiling_data_list = new_profiling_data_list

        # Normalize profiling data to compute avg_temp_after from thermal zones
        profiling_data_list = [normalize_profiling_data(d) for d in profiling_data_list]

        # -------------------------------------------------------------
        # Process results
        # -------------------------------------------------------------
        makespans = [d.get('time_elapsed', 0.0) for d in profiling_data_list]
        total_makespan = sum(makespans)
        energies = [d.get('total_energy_consumption', 0.0) for d in profiling_data_list]
        total_energy = sum(energies)

        temperatures = []
        for d in profiling_data_list:
            temp = d.get('avg_temp_after', None)
            if temp is not None:
                temperatures.append(temp)
            else:
                logging.warning(f"Missing avg_temp_after in profiling data for app {d.get('application_id')}")
                temperatures.append(target_temp)
        avg_temperature = float(np.mean(temperatures)) if temperatures else float(target_temp)

        logging.info(
            f"Thermal data: temperatures={[f'{t:.2f}' for t in temperatures]}, "
            f"avg={avg_temperature:.2f}°C"
        )

        # Per-iteration aggregated targets, for logging only
        target_makespans = [
            d.get('makespan_all_cores_frequency_11', 0.0) or 0.0 for d in profiling_data_list
        ]
        total_target_makespan = sum(target_makespans)
        if total_target_makespan <= 0:
            total_target_makespan = global_target_makespan

        target_energies = [
            d.get('energy_all_cores_frequency_0', 0.0) or 0.0 for d in profiling_data_list
        ]
        total_target_energy = sum(target_energies)
        if total_target_energy <= 0:
            total_target_energy = global_target_energy

        logging.info(
            f"Results: Makespan={total_makespan:.3f}s (target={total_target_makespan:.3f}s), "
            f"Energy={total_energy:.2f}J, Temp={avg_temperature:.1f}°C"
        )

        for app_idx, d in enumerate(profiling_data_list):
            makespan_per_app[app_idx].append(d.get('time_elapsed', 0.0))
            priority_per_app[app_idx].append(d.get('parallelism_level', 0.0) or 0.0)
            energy_per_app[app_idx].append(d.get('total_energy_consumption', 0.0))
            temperature_per_app[app_idx].append(d.get('avg_temp_after', target_temp))  # Track temp per app
            branch_misses_per_app[app_idx].append(d.get('branch_misses', 0))
            cache_misses_per_app[app_idx].append(d.get('cache_misses', 0))
            freq_per_app[app_idx].append(freq_step)
            cores_per_app[app_idx].append(applications_run[app_idx]['num_cores'])

        total_makespan_data.append(total_makespan)
        total_energy_data.append(total_energy)
        total_branch_misses_data.append(sum(d.get('branch_misses', 0) or 0 for d in profiling_data_list))
        total_cache_misses_data.append(sum(d.get('cache_misses', 0) or 0 for d in profiling_data_list))
        thermal_data.append(avg_temperature)  # Accumulate temperature history
        num_cores_data.append(num_cores)  # Track number of cores for standardized plotting

        # -------------------------------------------------------------
        # REVISED REWARDS with best_makespan tracking
        # -------------------------------------------------------------
        profiler_reward, got_improvement = get_reward_profiler(
            total_makespan,
            total_energy,
            global_target_makespan,
            global_target_energy,
            beta=beta,
            best_makespan=perf_tracker.best_makespan,
            iteration=t
        )
        
        thermal_reward = get_reward_thermal(avg_temperature, avg_temperature_prev, target_temp)

        # Total reward: performance-first + weighted thermal
        total_reward = profiler_reward + thermal_weight * thermal_reward

        logging.info("=" * 60)
        logging.info(f"Reward Breakdown (Iteration {t + 1}):")
        logging.info(f"  Profiler:  {profiler_reward:.4f}  (adaptive with improvement bonuses)")
        logging.info(f"  Thermal:   {thermal_reward:.4f}  (0 if T≤{target_temp}°C, negative if T>{target_temp}°C)")
        logging.info(f"  Total:     {total_reward:.4f}  (R_total = R_prof + λ_th * R_temp, λ_th={thermal_weight})")
        if got_improvement:
            logging.info(f"  ✅ IMPROVEMENT DETECTED! This is a new best!")
        logging.info("=" * 60)

        profiler_rewards.append(profiler_reward)
        thermal_rewards.append(thermal_reward)
        total_rewards.append(total_reward)
        priority_data.append(priority_tuple_int)

        # NEW: Update performance tracker
        improved = perf_tracker.update(total_makespan, total_energy, avg_temperature)

        stats_data['makespan'].append(total_makespan)
        stats_data['energy'].append(total_energy)
        stats_data['temperature'].append(avg_temperature)
        stats_data['profiler_reward'].append(profiler_reward)
        stats_data['thermal_reward'].append(thermal_reward)
        stats_data['total_reward'].append(total_reward)
        stats_data['best_makespan'].append(perf_tracker.best_makespan)
        stats_data['improvements'].append(1 if improved else 0)

        # -------------------------------------------------------------
        # Next states
        # -------------------------------------------------------------
        makespans_new = [pd_.get('time_elapsed', 0.0) for pd_ in profiling_data_list]
        total_mk_new = sum(makespans_new)
        new_priority_state = get_priority_agent_state(makespans_new, total_mk_new)
        new_thermal_state = get_thermal_agent_state(profiling_data_list)
        new_profiler_state = get_profiler_agent_state(profiling_data_list)
        if new_profiler_state is None:
            new_profiler_state = profiler_state.copy()

        # Profiler and thermal agents use their own rewards
        profiler_memory.store_observations(profiler_state, profiler_action_id, profiler_reward, new_profiler_state)
        thermal_memory.store_observations(thermal_state, thermal_action_id, thermal_reward, new_thermal_state)

        # Priority agent uses combined total reward (profiler + thermal_weight * thermal)
        priority_memory.store_observations(priority_state, priority_action_id, total_reward, new_priority_state)

        avg_temperature_prev = avg_temperature

        # -------------------------------------------------------------
        # Q-values & Loss Tracking
        # -------------------------------------------------------------
        if profiler_memory.initialized:
            p_input = profiler_memory.max_normalize(
                np.array([profiler_state], dtype=np.float32), mode='state'
            )
        else:
            p_input = np.array([profiler_state], dtype=np.float32)

        if thermal_memory.initialized:
            t_input = thermal_memory.max_normalize(
                np.array([thermal_state], dtype=np.float32), mode='state'
            )
        else:
            t_input = np.array([thermal_state], dtype=np.float32)

        if priority_memory.initialized:
            pr_input = priority_memory.max_normalize(
                np.array([priority_state], dtype=np.float32), mode='state'
            )
        else:
            pr_input = np.array([priority_state], dtype=np.float32)

        p_qval = profiler_agent.model.predict(p_input, verbose=0)[0]
        t_qval = thermal_agent.model.predict(t_input, verbose=0)[0]
        pr_qval = priority_agent.model.predict(pr_input, verbose=0)[0]

        profiler_qmax = float(np.amax(p_qval))
        thermal_qmax = float(np.amax(t_qval))
        priority_qmax = float(np.amax(pr_qval))

        profiler_qmax_data.append(profiler_qmax)
        thermal_qmax_data.append(thermal_qmax)
        priority_qmax_data.append(priority_qmax)

        profiler_loss_data.append(profiler_agent.currentLoss)
        thermal_loss_data.append(thermal_agent.currentLoss)
        priority_loss_data.append(priority_agent.currentLoss)

        stats_data['profiler_qmax'].append(profiler_qmax)
        stats_data['thermal_qmax'].append(thermal_qmax)
        stats_data['priority_qmax'].append(priority_qmax)
        stats_data['profiler_loss'].append(profiler_agent.currentLoss)
        stats_data['thermal_loss'].append(thermal_agent.currentLoss)
        stats_data['priority_loss'].append(priority_agent.currentLoss)

        ts.append(t)

        # -------------------------------------------------------------
        # CSV Data
        # -------------------------------------------------------------
        time_stamp = time.time()
        for pd_ in profiling_data_list:
            data_entry = {
                'timestamp': time_stamp,
                'run_mode': 'parallel',
                'num_cores': num_cores,
                'priority_tuple': priority_tuple_int,
                'iteration': t,
                'application_id': pd_.get('application_id'),
                'benchmark': pd_.get('benchmark'),
                'variant': pd_.get('variant'),
                'priority': pd_.get('priority'),
                'time_elapsed': pd_.get('time_elapsed'),
                'total_energy_consumption': pd_.get('total_energy_consumption'),
                'branch_misses': pd_.get('branch_misses'),
                'cache_misses': pd_.get('cache_misses'),
                'cores': pd_.get('cores'),
                'freq': freq_step,
                'makespan': pd_.get('time_elapsed'),
                'energy': pd_.get('total_energy_consumption'),
                'qmax': max(profiler_qmax, thermal_qmax, priority_qmax),
                'loss': (profiler_agent.currentLoss + thermal_agent.currentLoss + priority_agent.currentLoss) / 3.0,
                'reward': total_reward,
                'thermal': avg_temperature,
                'priority_combination': priority_tuple_int,
                'best_makespan': perf_tracker.best_makespan,
                'improved': 1 if improved else 0,
                'epsilon': profiler_agent.epsilon
            }
            for key, value in pd_.items():
                if key not in data_entry:
                    data_entry[key] = value
            collected_data.append(data_entry)

        # -------------------------------------------------------------
        # Plotting
        # -------------------------------------------------------------
        # Prepare current run data map for unified plotter
        current_run_data_map = {
            'makespan': total_makespan_data,
            'num_cores': num_cores_data,
            'cores': [np.mean([c[ti] for c in cores_per_app]) for ti in range(len(ts))] if ts else [],
            'qmax': [max(pq, tq, prq) for pq, tq, prq in zip(profiler_qmax_data, thermal_qmax_data, priority_qmax_data)],
            'energy': total_energy_data,
            'freq': [np.mean([f[ti] for f in freq_per_app]) for ti in range(len(ts))] if ts else [],
            'priority_combination': priority_data,
            'thermal': thermal_data,
            'branchmisses': total_branch_misses_data,
            'cachemisses': total_cache_misses_data,
            'priority': [np.mean(p) for p in zip(*priority_per_app)] if priority_per_app and priority_per_app[0] else [],
            'reward': total_rewards,
        }

        # Update plots using unified OnlinePlotter
        plotter.update(current_run_data_map, t)

        # -------------------------------------------------------------
        # Model-based training and agent updates
        # Uses model_train_start for environment model, agent_train_start for agents
        # Uses real_synthetic_ratio for batch composition
        # -------------------------------------------------------------
        if profiler_memory.mem_cntr >= model_train_start:
            logging.info("Training environment model for model-based learning...")
            env_model.train_model(profiler_memory, epochs=50)
            logging.info(f"Planning: generating {plan_count} synthetic samples...")
            env_model.long_horizon_reward_estimation(
                replay_buffer=profiler_memory,
                gen_buffer=gen_buffer,
                plan_count=plan_count,
                target_makespan=global_target_makespan,
                target_energy=global_target_energy,
                target_temp=target_temp,
                beta=beta,
                thermal_weight=thermal_weight,
                best_makespan=perf_tracker.best_makespan
            )
            logging.info(f"Model-based training complete (epsilon={profiler_agent.epsilon:.3f})")

            # Calculate batch sizes based on real_synthetic_ratio
            real_batch_size = int(batch_size * real_synthetic_ratio)
            synthetic_batch_size = batch_size - real_batch_size

            # Train with real data
            profiler_agent.train_model(profiler_memory.sample_observation(real_batch_size, random=True))
            thermal_agent.train_model(thermal_memory.sample_observation(real_batch_size, random=True))
            priority_agent.train_model(priority_memory.sample_observation(real_batch_size, random=True))

            # Train with synthetic data
            profiler_agent.train_model(profiler_gen_memory.sample_observation(synthetic_batch_size, random=True))
            thermal_agent.train_model(thermal_gen_memory.sample_observation(synthetic_batch_size, random=True))
            priority_agent.train_model(priority_gen_memory.sample_observation(synthetic_batch_size, random=True))

            profiler_agent.update_target_model()
            thermal_agent.update_target_model()
            priority_agent.update_target_model()

        # -------------------------------------------------------------
        # Save CSV and models periodically
        # -------------------------------------------------------------
        iterations_since_last_save += 1
        if iterations_since_last_save >= checkpoint_interval:
            df = pd.DataFrame(collected_data)
            df.to_csv(csv_filename, index=False)
            logging.info(f"Checkpoint: Saved CSV data to {csv_filename}")

            if save_model:
                model_dir = save_dir
                os.makedirs(model_dir, exist_ok=True)

                profiler_memory.save_data(os.path.join(model_dir, 'profiler_state_transitions.csv'))
                thermal_memory.save_data(os.path.join(model_dir, 'thermal_state_transitions.csv'))
                priority_memory.save_data(os.path.join(model_dir, 'priority_state_transitions.csv'))

                profiler_gen_memory.save_data(os.path.join(model_dir, 'profiler_gen_state_transitions.csv'))
                thermal_gen_memory.save_data(os.path.join(model_dir, 'thermal_gen_state_transitions.csv'))
                priority_gen_memory.save_data(os.path.join(model_dir, 'priority_gen_state_transitions.csv'))

                profiler_agent.model.save_weights(os.path.join(model_dir, "profiler_model_data.weights.h5"))
                thermal_agent.model.save_weights(os.path.join(model_dir, "thermal_model_data.weights.h5"))
                priority_agent.model.save_weights(os.path.join(model_dir, "priority_model_data.weights.h5"))

                logging.info("Checkpoint: Saved models and replay buffers")

            iterations_since_last_save = 0

        # -------------------------------------------------------------
        # Update current states
        # -------------------------------------------------------------
        profiler_state = new_profiler_state
        thermal_state = new_thermal_state
        priority_state = new_priority_state

    # -------------------------------------------------------------------------
    # Final save at the end of training
    # -------------------------------------------------------------------------
    df = pd.DataFrame(collected_data)
    df.to_csv(csv_filename, index=False)
    logging.info(f"Final: Saved CSV data to {csv_filename}")

    if save_model:
        model_dir = save_dir
        os.makedirs(model_dir, exist_ok=True)

        profiler_memory.save_data(os.path.join(model_dir, 'profiler_state_transitions.csv'))
        thermal_memory.save_data(os.path.join(model_dir, 'thermal_state_transitions.csv'))
        priority_memory.save_data(os.path.join(model_dir, 'priority_state_transitions.csv'))

        profiler_gen_memory.save_data(os.path.join(model_dir, 'profiler_gen_state_transitions.csv'))
        thermal_gen_memory.save_data(os.path.join(model_dir, 'thermal_gen_state_transitions.csv'))
        priority_gen_memory.save_data(os.path.join(model_dir, 'priority_gen_state_transitions.csv'))

        profiler_agent.model.save_weights(os.path.join(model_dir, "profiler_model_data.weights.h5"))
        thermal_agent.model.save_weights(os.path.join(model_dir, "thermal_model_data.weights.h5"))
        priority_agent.model.save_weights(os.path.join(model_dir, "priority_model_data.weights.h5"))

        logging.info("Final: Saved models and replay buffers")

    stats_csv = csv_filename.replace('.csv', '_stats.csv')
    export_statistics_to_csv(stats_data, stats_csv)

    # Save final plot using unified OnlinePlotter - derive from csv_filename for consistency
    try:
        plot_filename = os.path.basename(csv_filename).replace('.csv', '_plot.png')
        plot_path = os.path.join(save_dir, plot_filename)
        plotter.save(plot_path)
        logging.info(f"Final: Saved plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")

    plotter.close()  # Free memory

    # Print final statistics
    logging.info("\n" + "=" * 80)
    logging.info("TRAINING COMPLETE - FINAL STATISTICS")
    logging.info("=" * 80)
    perf_stats = perf_tracker.get_stats()
    logging.info(f"Best makespan achieved: {perf_stats['best_makespan']:.3f}s")
    logging.info(f"Best energy achieved: {perf_stats['best_energy']:.3f}J")
    logging.info(f"Total improvements: {perf_stats['total_improvements']}")
    logging.info(f"Target makespan: {global_target_makespan:.3f}s")
    if perf_stats['best_makespan'] < float('inf'):
        gap = (perf_stats['best_makespan'] / global_target_makespan - 1) * 100
        logging.info(f"Gap to target: {gap:.1f}%")
    logging.info("=" * 80)

    logging.info("Training completed.")

    # Return results for comparison with other modules (same format as MAMFRL)
    return ts, makespan_per_app, energy_per_app, temperature_per_app