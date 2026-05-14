#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAMBRL - Single-Agent Model-Based RL for Jetson TX2
FIXED VERSION - Compatible with combined_tx2_fixed.py

KEY FIXES FROM server_sambrl.py:
=================================
1. Renamed function from train_sambrl to train_fixed_sambrl
2. Hyperparameter-aware filename format: SAMBRL_YYYYMMDD_HHMMSS_EXPep_betaB_lrL_epsE_batchB.csv
3. Added helper functions: parse_hyperparams_from_filename(), load_historical_data(), extract_server_name()
4. Historical data loading with cross-module comparison (SAMBRL vs SAFRL with same experiment count)
5. One row per iteration (not per app) - aggregate per-app data into single iteration metrics
6. Online plotting with comparison to historical data
7. Return values: (ts, makespan_per_app, energy_per_app, temperature_per_app) for combined_tx2_fixed.py
8. FIXED: Proper temperature tracking with fallbacks
9. FIXED: Cores tracking as string for CSV and bitmask for plotting
10. FIXED: Branch misses and cache misses properly tracked and plotted

SAMBRL ARCHITECTURE:
====================
- Single profiler agent (no temperature/priority agents)
- Profiler agent WITH environment model for planning (model-based)
- Environment model predicts next state from (state, action) pairs
- Dyna-Q style: learns from both real experiences and synthetic rollouts
- Reward: combines makespan and energy with beta weighting
"""

import logging
import json
import time
import numpy as np
import random
import csv
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import re
import socket
import threading
import argparse
from datetime import datetime
from scipy import stats
import torch as T

# Import unified plotting module
from live_plotter import (
    OnlinePlotter, create_data_map,
    load_historical_data, extract_server_name, parse_hyperparams_from_filename
)
from server_combined import DATA_KEYS
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of this script for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "save_model")

STATE_SIZE = None


###############################################################################
# Helper to convert a "core subset string" (e.g., "1,2") to a bitmask integer
###############################################################################
def subset_to_int(subset_str: str) -> int:
    subset_int = 0
    cores = subset_str.split(',')
    for core in cores:
        core = core.strip()
        if core.isdigit():
            core_id = int(core)
            subset_int |= (1 << core_id)
    return subset_int


###############################################################################
# Normalize profiling data to ensure all expected keys exist
# Maps new client_evaluate_tx2.py format to expected state format
###############################################################################
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
            # Fallback to old keys
            for key in ['CPU57_temp_after', 'MCPU_temp', 'CPU_temp']:
                if data.get(key) is not None and data.get(key, 0) > 0:
                    data['avg_temp_after'] = float(data.get(key))
                    break
            else:
                data['avg_temp_after'] = 50.0  # Default

    # --- Map energy keys: new format -> old format ---
    # New: energy_cpu_j, energy_denver_j, energy_system_j, energy_gpu_j, energy_ddr_j
    # Old: CPU57_energy_joules, Denver_energy_joules, System_energy_joules, GPU_energy_joules, DDR_energy_joules
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
    # TX2 thermal zones typically: zone0=CPU, zone1=GPU, zone2=AUX, etc.
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


###############################################################################
# State parser from profiling data
###############################################################################
def parse_state(profiling_data):
    global STATE_SIZE
    try:
        # Normalize data to ensure all expected keys exist
        profiling_data = normalize_profiling_data(profiling_data)

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

###############################################################################
# Convert a priority tuple to a single int (e.g., (10,10,10) -> 101010)
###############################################################################
def priority_tuple_to_int(tup):
    return int(''.join(str(x) for x in tup))


# --- Revised Reward Function ---
def get_reward_profiler(makespan, avg_energy_consumption, target_makespan, target_energy, beta=0.5):
    # Invert the ratios so that lower makespan/energy (better performance) gives a higher reward.
    epsilon=1e-3
    if makespan <= epsilon:
        makespan = target_makespan
    if avg_energy_consumption <= epsilon:
        avg_energy_consumption = target_energy
    makespan_reward = target_makespan / (makespan + epsilon)
    energy_reward = target_energy / (avg_energy_consumption + epsilon)
    reward_value = beta * makespan_reward + (1 - beta) * energy_reward
    logging.debug(f"Target makespan was {target_makespan} and makespan was {makespan}. Target energy was {target_energy} and energy was {avg_energy_consumption}. Makespan reward: {makespan_reward}, Energy reward: {energy_reward}, Combined reward: {reward_value}")
    return reward_value


###############################################################################
# Replay Buffer
###############################################################################
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
        """
        Normalizes the input array based on min/max found in memory.
        For actions, we use action_size as max.
        """
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

###############################################################################
# DQN Agent
###############################################################################
class DQNAgent:
    def __init__(
        self, state_size, action_size, load_model, discount_factor,
        epsilon, epsilon_decay, epsilon_min, epsilon_start, epsilon_end,
        batch_size, memsize, learning_rate, memory=None
    ):
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
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        # Convert single state to batch
        state = np.array([state], dtype=np.float32)
        # Normalize if memory is initialized
        if self.memory and self.memory.initialized:
            state_norm = self.memory.max_normalize(state, mode='state')
        else:
            state_norm = state

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

        if self.memory and self.memory.initialized:
            states_norm = self.memory.max_normalize(states, mode='state')
            next_states_norm = self.memory.max_normalize(next_states, mode='state')
        else:
            states_norm = states
            next_states_norm = next_states

        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        target = self.model.predict(states_norm, verbose=0)
        target_val = self.target_model.predict(next_states_norm, verbose=0)

        for i in range(self.batch_size):
            a = actions[i]
            target[i][a] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        hist = self.model.fit(states_norm, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.currentLoss = hist.history['loss'][0]

###############################################################################
# Environment Model (PyTorch-based)
###############################################################################
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
    def __init__(self, input_dim, output_dim, batch_size):
        super().__init__()
        learning_rate = 0.001
        self.model = FCNModel(input_dim, 128, output_dim)
        self.state_size = output_dim

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size

    def train_model(self, replay_buffer, epochs=100):
        """
        Train the environment model on real samples from replay_buffer.
        """
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

            epoch_loss = 0
            for inp, targets in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(inp)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            logging.info(f'Epoch {epoch + 1}/{epochs}: Average Loss = {avg_loss:.4f}')

    def planning(self, replay_buffer, gen_buffer, num_samples, target_makespan=30, target_energy=50, beta=2):
        """
        Use the environment model to sample next states from random states & actions,
        storing the resulting transitions into gen_buffer for agent training.
        """
        # Reset gen buffer's memory counter so it acts like a new set
        gen_buffer['profiler'].mem_cntr = 0

        for _ in range(num_samples):
            one_observation = replay_buffer.sample_observation(batch_size=1, random=True)
            if one_observation is None:
                continue

            state = np.array([obs[0] for obs in one_observation], dtype=np.float32).flatten()
            action = np.array([obs[1] for obs in one_observation], dtype=np.float32).flatten()

            normal_state = replay_buffer.max_normalize(state, mode='state')
            normal_action = replay_buffer.max_normalize(action, mode='action')
            normal_state_tensor = T.tensor(normal_state, dtype=T.float32).unsqueeze(0)
            normal_action_tensor = T.tensor(normal_action, dtype=T.float32).unsqueeze(0)

            normal_input_tensor = T.cat([normal_state_tensor, normal_action_tensor], dim=1)
            normal_next_state_tensor = self.model(normal_input_tensor)
            denormal_next_state = replay_buffer.max_denormalize(
                normal_next_state_tensor.detach().cpu().numpy(), mode='state'
            )[0]

            denormal_state = state
            denormal_action = action

            makespan = denormal_next_state[16]
            avg_energy_consumption = np.mean(denormal_next_state[17:22])

            profiler_reward = get_reward_profiler(
                makespan=makespan,
                avg_energy_consumption=avg_energy_consumption,
                target_makespan=target_makespan,
                target_energy=target_energy,
                beta=beta
            )

            gen_buffer['profiler'].store_observations(
                state=denormal_state,
                action=int(denormal_action[0]),
                reward=profiler_reward,
                state_=denormal_next_state
            )

###############################################################################
# Get single "profiler state" from first application in a batch
###############################################################################
def get_profiler_agent_state(applications_data):
    if not applications_data:
        return None
    parsed = parse_state(applications_data[0])
    if parsed is None:
        return None
    state, _, _, _, _, _, _, _, _ = parsed
    return state


###############################################################################
# Helper function to get temperature with fallbacks
###############################################################################
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


###############################################################################
# The main training function for SAMBRL (FIXED VERSION)
###############################################################################
def train_fixed_sambrl(
    client_socket,
    data_keys=DATA_KEYS,
    experiment_time=100,
    clock_change_time=30,
    beta=1.0,
    load_model=False,
    learn_count=32,           # DEPRECATED: Use model_train_start instead
    plan_count=100,           # Synthetic samples per planning phase
    mem_size=100000,
    learning_rate=0.1,        # OPTIMAL from tuning
    discount_factor=0.99,
    epsilon=1,
    epsilon_decay=0.90,
    epsilon_min=0.10,         # OPTIMAL from tuning
    epsilon_start=1.0,
    epsilon_end=0.01,
    reset_learning_rate_value=20,
    save_repetition=50,
    save_model=True,
    batch_size=64,            # OPTIMAL from tuning
    agent_train_start=32,     # Total data threshold for agent training
    target_temp=50,
    server_name_1='',
    server_name_2='',
    server_name_main='sambrl_main.csv',
    profiling_data_list=None,
    application_profiles=None,
    applications_fixed=None,
    priority_combinations=None,
    frequency_combinations = [[0]*5,[2]*5,[6]*5,[11]*5],
    num_cores_list = [1,3,5],
    # New model-based RL parameters
    model_train_start=32,     # Real data threshold for model training
    real_synthetic_ratio=0.5  # 50% real, 50% synthetic in training batch
):
    """
    SAMBRL FIXED - Single-Agent Model-Based RL with hyperparameter-aware naming.

    MODEL-BASED RL PARAMETERS:
    - plan_count: Number of synthetic samples generated per planning phase
    - model_train_start: Real data threshold before environment model training starts
    - agent_train_start: Total data threshold before agent training starts
    - real_synthetic_ratio: Ratio of real to synthetic data in training batch (0.5 = balanced)

    Returns:
        (ts, makespan_per_app, energy_per_app, temperature_per_app) for combined_tx2_fixed.py
    """
    # Use model_train_start if provided, otherwise fall back to learn_count
    if model_train_start is None:
        model_train_start = learn_count
    logging.info("=" * 80)
    logging.info("SAMBRL FIXED - Single-Agent Model-Based RL")
    logging.info(f"Experiment time: {experiment_time} episodes")
    logging.info(f"Beta (makespan vs energy): {beta}")
    logging.info("-" * 40)
    logging.info("MODEL-BASED RL PARAMETERS:")
    logging.info(f"  Plan count (synthetic data)  : {plan_count}")
    logging.info(f"  Model train start (real data): {model_train_start}")
    logging.info(f"  Agent train start (total)    : {agent_train_start}")
    logging.info(f"  Real/Synthetic ratio         : {real_synthetic_ratio}")
    logging.info("-" * 40)
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Epsilon min: {epsilon_min}")
    logging.info(f"Batch size: {batch_size}")
    logging.info("=" * 80)

    # Basic checks
    if profiling_data_list is None:
        profiling_data_list = []
    if applications_fixed is None:
        applications_fixed = []
    if len(applications_fixed) == 0:
        logging.error("No applications.")
        return ([], [[] for _ in range(3)], [[] for _ in range(3)], [[] for _ in range(3)])

    num_apps = len(applications_fixed)
    fixed_priority_tuple = tuple([10]*num_apps)

    # Create possible (freq,core) actions
    profiler_actions = []
    for freq_combo in frequency_combinations:
        for nc in num_cores_list:
            profiler_actions.append((freq_combo[0], nc))
    profiler_action_space = len(profiler_actions)

    # Check we have some initial data
    if not profiling_data_list:
        logging.error("No initial profiling data available.")
        return ([], [[] for _ in range(num_apps)], [[] for _ in range(num_apps)], [[] for _ in range(num_apps)])

    # Parse the initial state
    profiler_state = get_profiler_agent_state(profiling_data_list)
    if profiler_state is None:
        logging.error("Could not parse initial profiler state.")
        return ([], [[] for _ in range(num_apps)], [[] for _ in range(num_apps)], [[] for _ in range(num_apps)])

    # Real & "generated" memories
    profiler_memory = ReplayBuffer(mem_size, len(profiler_state), profiler_action_space)
    profiler_gen_memory = ReplayBuffer(mem_size, len(profiler_state), profiler_action_space)

    # Initialize the agent
    profiler_agent = DQNAgent(
        state_size=len(profiler_state),
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
        learning_rate=learning_rate,
        memory=profiler_memory
    )

    # Environment model: input_dim = state + 1 action, output_dim = next_state
    input_dim = len(profiler_state) + 1
    output_dim = len(profiler_state)
    env_model = EnvironmentModel(input_dim, output_dim, batch_size)

    # Use server_name_main if provided (from server_combined.py), otherwise generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if server_name_main and server_name_main != 'sambrl_main.csv':
        csv_filename = server_name_main if os.path.isabs(server_name_main) else os.path.join(save_dir, server_name_main)
    else:
        csv_filename = os.path.join(save_dir, f"SAMBRL_{timestamp}_{experiment_time}ep_beta{beta}_lr{learning_rate}_eps{epsilon_min}_batch{batch_size}.csv")

    logging.info(f"Output CSV: {csv_filename}")

    # If loading model from disk
    if load_model:
        profiler_memory.load_data(os.path.join(save_dir, "sambrl_profiler_state_transitions.csv.npz"))
        profiler_gen_memory.load_data(os.path.join(save_dir, "sambrl_profiler_gen_state_transitions.csv.npz"))
        logging.info(f"Loaded real state transitions {profiler_memory.mem_cntr} and generator states {profiler_gen_memory.mem_cntr}")

        model_path = os.path.join(save_dir, "sambrl_profiler_model_data.weights.h5")
        if os.path.exists(model_path):
            profiler_agent.model.load_weights(model_path)


    # Prepare lists to track metrics
    ts = []
    makespan_per_app = [[] for _ in applications_fixed]
    priority_per_app = [[] for _ in applications_fixed]
    energy_per_app = [[] for _ in applications_fixed]
    temperature_per_app = [[] for _ in applications_fixed]
    branch_misses_per_app = [[] for _ in applications_fixed]
    cache_misses_per_app = [[] for _ in applications_fixed]
    freq_per_app = [[] for _ in applications_fixed]
    cores_per_app = [[] for _ in applications_fixed]

    # Aggregated per-iteration metrics (one row per iteration)
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
    cores_combination_data = []  # Bitmask of cores used
    cores_str_data = []  # String representation of cores used (e.g., "1,2,3")
    num_cores_data = []
    freq_data = []  # Added for proper frequency tracking
    priority_values_data = []  # Added for proper priority tracking
    reward_sum = 0.0
    count_steps = 0
    avg_temperature_prev = target_temp

    CoreList = [1,2,3,4,5]
    priority_tuple_int = priority_tuple_to_int(fixed_priority_tuple)

    collected_data = []
    iterations_since_last_save = 0
    checkpoint_interval = save_repetition

    # Setup real-time plotting using unified OnlinePlotter
    plotter = OnlinePlotter(
        data_keys=data_keys,
        experiment_time=experiment_time,
        module_name="SAMBRL",
        save_dir=save_dir,
        server_name_1=server_name_1,
        server_name_2=server_name_2,
        baseline_label="SAMFRL (Model-Free)",
        previous_label="SAMBRL (Previous)"
    )

    gen_buffer = {
        'profiler': profiler_gen_memory
    }

    # Main loop
    for t in range(experiment_time):
        # Decide an action with the profiler agent
        profiler_action_id = profiler_agent.get_action(profiler_state)
        freq_step, num_cores = profiler_actions[profiler_action_id]

        current_priority_tuple = fixed_priority_tuple
        num_cores_data.append(num_cores)
        freq_data.append(freq_step)

        chosen_freq_comb = [freq_step]*5
        available_cores = CoreList[:num_cores]

        # Build the run instructions for each application
        applications_run = []
        for app_idx, app in enumerate(applications_fixed):
            cores_assigned = available_cores.copy()
            cores_assigned_str = ','.join(map(str, cores_assigned))
            frequencies_assigned = chosen_freq_comb[:len(cores_assigned)]
            # FIXED: Include all fields that client expects (benchmark, variant, input_arg, app_args)
            app_instance = {
                'id': app['id'],
                'benchmark': app.get('benchmark', app.get('path', '')),
                'variant': app.get('variant', ''),
                'input_arg': app.get('input_arg', ''),
                'app_args': app.get('app_args', app.get('input_arg', '')),
                'path': app.get('path', ''),
                'priority': fixed_priority_tuple[app_idx],
                'frequencies': frequencies_assigned,
                'num_cores': len(cores_assigned),
                'cores': cores_assigned_str,
                'action': 'run'
            }
            applications_run.append(app_instance)

        logging.info(f"[Episode {t}] Freq {freq_step}, Cores {num_cores}, Priority {fixed_priority_tuple}, cores_assigned {cores_assigned_str}, replay_buffer_count {profiler_memory.mem_cntr}")

        # Send the JSON message
        send_msg_dict = {
            'applications': applications_run,
            'run_mode': 'parallel'
        }
        send_msg = json.dumps(send_msg_dict)
        client_socket.send((send_msg + '\n').encode())
        logging.info("Sent apps to client.")

        # Wait for the client to return the profiling data
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

        # Compute union of core combinations from profiling data (as bitmask and string)
        union_cores = 0
        all_cores_set = set()
        for d in profiling_data_list:
            cores_str = d.get('cores', '')
            union_cores |= subset_to_int(cores_str)
            # Collect all unique cores used
            if cores_str:
                for c in cores_str.split(','):
                    c = c.strip()
                    if c.isdigit():
                        all_cores_set.add(int(c))
        cores_combination_data.append(union_cores)
        # Store cores as sorted string (e.g., "1,2,3")
        cores_str_data.append(','.join(map(str, sorted(all_cores_set))) if all_cores_set else '')


        # Summaries
        makespans = [d.get('time_elapsed',0.0) for d in profiling_data_list]
        total_makespan = sum(makespans)
        energies = [d.get('total_energy_consumption',0.0) for d in profiling_data_list]
        total_energy = sum(energies)

        # Get temperature with fallbacks
        temps = [get_temperature_with_fallbacks(d, target_temp) for d in profiling_data_list]
        avg_temperature = np.mean(temps) if temps else target_temp
        thermal_data.append(avg_temperature)

        # Get branch misses and cache misses
        total_branch_misses = sum(d.get('branch_misses', 0) or 0 for d in profiling_data_list)
        total_cache_misses = sum(d.get('cache_misses', 0) or 0 for d in profiling_data_list)
        total_branch_misses_data.append(total_branch_misses)
        total_cache_misses_data.append(total_cache_misses)

        # Some "target" values from the data
        target_makespans = [
            d.get('makespan_all_cores_frequency_11', 0.0)
            for d in profiling_data_list if d.get('makespan_all_cores_frequency_11') is not None
        ]
        total_target_makespan = sum(target_makespans) if target_makespans else total_makespan
        target_energies = [
            d.get('energy_all_cores_frequency_0', 0.0)
            for d in profiling_data_list if d.get('energy_all_cores_frequency_0') is not None
        ]
        total_target_energy = sum(target_energies) if target_energies else total_energy

        # Per-app stats
        for app_idx, d in enumerate(profiling_data_list):
            mk = d.get('time_elapsed',0.0)
            p_lev = d.get('parallelism_level',0.0)
            if p_lev is None:
                p_lev = 0.0
            makespan_per_app[app_idx].append(mk)
            priority_per_app[app_idx].append(fixed_priority_tuple[app_idx])
            en = d.get('total_energy_consumption',0.0)
            energy_per_app[app_idx].append(en)

            # Get temperature with fallbacks
            temp = get_temperature_with_fallbacks(d, target_temp)
            temperature_per_app[app_idx].append(temp)

            bmiss = d.get('branch_misses', 0) or 0
            cmiss = d.get('cache_misses', 0) or 0
            branch_misses_per_app[app_idx].append(bmiss)
            cache_misses_per_app[app_idx].append(cmiss)
            freq_per_app[app_idx].append(freq_step)
            app_inst = applications_run[app_idx]
            cores_count = app_inst['num_cores']
            cores_per_app[app_idx].append(cores_count)

        total_makespan_data.append(total_makespan)
        total_energy_data.append(total_energy)

        # Compute reward
        profiler_reward = get_reward_profiler(
            total_makespan, total_energy,
            total_target_makespan, total_target_energy,
            beta=beta
        )
        total_reward = profiler_reward

        reward_sum += total_reward
        count_steps += 1
        profiler_rewards.append(profiler_reward)
        total_rewards.append(total_reward)
        priority_data.append(priority_tuple_int)
        priority_values_data.append(np.mean(fixed_priority_tuple))

        # Next state
        new_profiler_state = get_profiler_agent_state(profiling_data_list)
        if new_profiler_state is None:
            new_profiler_state = profiler_state.copy()

        # Store the real observation
        profiler_memory.store_observations(profiler_state, profiler_action_id, profiler_reward, new_profiler_state)

        avg_temperature_prev = avg_temperature

        # Compute Q-values from the current state
        if profiler_memory.initialized:
            st_norm = profiler_memory.max_normalize(np.array([profiler_state], dtype=np.float32), mode='state')
            p_qval = profiler_agent.model.predict(st_norm, verbose=0)[0]
        else:
            p_qval = profiler_agent.model.predict(np.array([profiler_state], dtype=np.float32), verbose=0)[0]

        profiler_qmax = np.amax(p_qval)
        profiler_qmax_data.append(profiler_qmax)
        profiler_loss_data.append(profiler_agent.currentLoss)
        ts.append(t)

        # ONE ROW PER ITERATION (aggregate per-app data)
        iteration_entry = {
            'iteration': t,
            'run_mode': 'parallel',
            'num_cores': num_cores,
            'priority_tuple': priority_tuple_int,
            'makespan': total_makespan,
            'energy': total_energy,
            'thermal': avg_temperature,
            'branchmisses': total_branch_misses,
            'cachemisses': total_cache_misses,
            'qmax': profiler_qmax,
            'loss': profiler_agent.currentLoss,
            'reward': total_reward,
            'freq': freq_step,
            'cores': cores_str_data[-1],  # Use string representation for CSV readability
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
                if col_name not in iteration_entry:
                    iteration_entry[col_name] = v

        # Also save aggregated thermal zone data across all apps
        if profiling_data_list:
            # Get thermal zones from first app (they should be system-wide)
            first_d = profiling_data_list[0]
            for k, v in first_d.items():
                if k.startswith('thermal_zone') or k.startswith('energy_') or k.startswith('power_'):
                    if k not in iteration_entry:
                        iteration_entry[k] = v

        collected_data.append(iteration_entry)

        # Prepare data for plotting using unified create_data_map
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

        # Agent training using both real and generated data (MODEL-BASED PLANNING)
        # Use model_train_start to control when environment model training begins
        if profiler_memory.mem_cntr >= model_train_start:  # Real data threshold check
            if t % model_train_start == 0 or t == 0:
                logging.info("Training the environment model...")
                env_model.train_model(replay_buffer=profiler_memory, epochs=100)
                logging.info(f"Planning: generating {plan_count} synthetic samples...")
                env_model.planning(
                    replay_buffer=profiler_memory,
                    gen_buffer=gen_buffer,
                    num_samples=plan_count,
                    target_makespan=total_target_makespan,
                    target_energy=total_target_energy,
                    beta=beta
                )

                # Calculate batch sizes based on real_synthetic_ratio
                # real_synthetic_ratio = 0.5 means 50% real, 50% synthetic
                real_batch_size = int(batch_size * real_synthetic_ratio)
                synthetic_batch_size = batch_size - real_batch_size

                profiler_real_batch = profiler_memory.sample_observation(batch_size=real_batch_size, random=True)
                profiler_gen_batch = profiler_gen_memory.sample_observation(batch_size=synthetic_batch_size, random=True)

                # Combine real and generated experiences based on ratio
                if profiler_real_batch and profiler_gen_batch:
                    profiler_combined_batch = profiler_real_batch + profiler_gen_batch
                elif profiler_real_batch:
                    profiler_combined_batch = profiler_real_batch
                else:
                    profiler_combined_batch = None

                if profiler_combined_batch:
                    profiler_agent.train_model(memory=profiler_combined_batch)
                if t % save_repetition == 0:
                    profiler_agent.update_target_model()

        # Print Q-max each iteration
        logging.info(f"[Episode {t}] Q-max: {profiler_qmax:.4f}, Reward: {total_reward:.4f}, Epsilon: {profiler_agent.epsilon:.4f}, Temp: {avg_temperature:.1f}C, BranchMiss: {total_branch_misses}, CacheMiss: {total_cache_misses}")

        # Periodic saving
        iterations_since_last_save += 1
        if iterations_since_last_save >= checkpoint_interval and checkpoint_interval != 0:
            if save_model:
                os.makedirs(save_dir, exist_ok=True)
                profiler_agent.model.save_weights(os.path.join(save_dir, "sambrl_profiler_model_data.weights.h5"))
                profiler_memory.save_data(os.path.join(save_dir, "sambrl_profiler_state_transitions.csv"))
                profiler_gen_memory.save_data(os.path.join(save_dir, "sambrl_profiler_gen_state_transitions.csv"))
                logging.info("[Save models and replay memories]")

            if collected_data:
                csv_path = os.path.join(save_dir, csv_filename)
                df_csv = pd.DataFrame(collected_data)
                df_csv.to_csv(csv_path, index=False)
                logging.info(f"Saved collected data to '{csv_path}'.")

            iterations_since_last_save = 0

        # Move on
        profiler_state = new_profiler_state

    # Final data save
    if collected_data:
        csv_path = os.path.join(save_dir, csv_filename)
        df_csv = pd.DataFrame(collected_data)
        df_csv.to_csv(csv_path, index=False)
        logging.info(f"Saved collected data to '{csv_path}'.")

    # Save final plot using unified plotter - derive from csv_filename for consistency
    plot_filename = os.path.basename(csv_filename).replace('.csv', '_plot.png')
    plotter.save(os.path.join(save_dir, plot_filename))
    plotter.close()  # Free memory

    logging.info("SAMBRL training completed.")

    # Return data for combined_tx2_fixed.py
    return (ts, makespan_per_app, energy_per_app, temperature_per_app)
