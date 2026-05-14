#!/usr/bin/env python3
"""
graphperf_world_model_experiment.py - GraphPerf-RT as World Model for RL

Replaces FCNModel+EnvModel in MAMBRL with GraphPerf-RT GAT predictor.
Compares:
1. MAMBRL with original FCN environment model
2. MAMBRL with GraphPerf-RT GAT world model
3. MAMFRL (model-free baseline)

Based on zero_combined_twobench.py and zero_mambrl.py patterns from ZeroDVFS.

Purpose:
- Show GraphPerf-RT can replace traditional environment models in model-based RL
- Compare prediction accuracy for planning (future reward estimation)
- Measure convergence speed improvement

Usage:
    python graphperf_world_model_experiment.py \
        --model ../GAT_EFG-new_training/checkpoints/best_model.pt \
        --config ../GAT_EFG-new_training/configs/model_config_v2.yaml \
        --epochs 200 \
        --seeds 42 123 456
"""

import argparse
import gc
import json
import logging
import os
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup paths
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir.parent / 'GAT_EFG-new_training'))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_KEYS = [
    'makespan', 'num_cores', 'energy', 'qmax', 'loss', 'freq', 'cores',
    'thermal', 'reward', 'branch_misses', 'cache_misses', 'priority_combination'
]

# Default hyperparameters (from zero_combined_twobench.py)
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
}


# =============================================================================
# GRAPHPERF-RT WORLD MODEL
# =============================================================================
class GraphPerfWorldModel:
    """
    Wraps GraphPerf-RT GAT as world model for MAMBRL.

    Replaces FCNModel+EnvModel from ZeroDVFS with our heterogeneous GAT.
    Interface matches zero_sarbrl_v3.py environment model API.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = 'cuda'
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for GraphPerf-RT world model")

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logging.info(f"GraphPerf-RT World Model using device: {self.device}")

        # Load config
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load GAT model
        try:
            from src.models.gat import create_model
            self.model = create_model(self.config['model']).to(self.device)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.eval()
            logging.info(f"Loaded GraphPerf-RT model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

        # Prediction statistics
        self.prediction_count = 0
        self.prediction_errors = []

    def state_to_graph(
        self,
        state: np.ndarray,
        action: int,
        num_cores: int = 6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert RL state + action to graph representation.

        Args:
            state: RL state vector from parse_state()
            action: Frequency action (0-11)
            num_cores: Number of CPU cores

        Returns:
            node_features, edge_index, global_features
        """
        # Node features: [core_active, freq_level, freq_hz]
        # Based on GAT_EFG-new_training/src/data/dataset.py
        node_features = []

        # Extract core info from state (subset_int is at index 1)
        subset_int = int(state[1]) if len(state) > 1 else 0
        freq_level = action  # Action is the frequency index

        # Map frequency index to Hz (TX2 frequencies)
        freq_map = {
            0: 345600, 1: 499200, 2: 652800, 3: 806400,
            4: 960000, 5: 1113600, 6: 1267200, 7: 1420800,
            8: 1574400, 9: 1728000, 10: 1881600, 11: 2035200
        }
        freq_hz = freq_map.get(freq_level, 2035200)

        for core_id in range(num_cores):
            core_active = 1.0 if (subset_int & (1 << core_id)) else 0.0
            node_features.append([
                core_active,
                freq_level / 11.0,  # Normalize
                freq_hz / 2500000.0  # Normalize
            ])

        node_features = torch.tensor(node_features, dtype=torch.float32)

        # Edge index: Fully connected graph
        edge_index = []
        for i in range(num_cores):
            for j in range(num_cores):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Global features from state
        # Extract relevant global features from state vector
        global_features = torch.tensor([
            state[2] if len(state) > 2 else 0,   # utilization
            state[16] if len(state) > 16 else 0, # time_elapsed
            state[17] if len(state) > 17 else 0, # energy
        ], dtype=torch.float32)

        return node_features, edge_index, global_features

    def predict(
        self,
        state: np.ndarray,
        action: int
    ) -> float:
        """
        Predict makespan for given state-action pair.

        Args:
            state: Current state
            action: Frequency action

        Returns:
            Predicted makespan (seconds)
        """
        self.prediction_count += 1

        with torch.no_grad():
            node_feat, edge_idx, global_feat = self.state_to_graph(state, action)

            # Move to device
            node_feat = node_feat.to(self.device)
            edge_idx = edge_idx.to(self.device)
            global_feat = global_feat.to(self.device)

            # Add batch dimension
            batch = torch.zeros(node_feat.size(0), dtype=torch.long, device=self.device)

            # Forward pass
            prediction = self.model(
                node_feat, edge_idx, global_feat.unsqueeze(0), batch
            )

            return prediction.item()

    def predict_next_state(
        self,
        state: np.ndarray,
        action: int
    ) -> np.ndarray:
        """
        Predict next state given current state and action.

        Matches interface: env_model.predict(state, action) from ZeroDVFS.
        """
        predicted_makespan = self.predict(state, action)

        # Create next state by modifying relevant components
        next_state = state.copy()
        next_state[0] = action  # Update frequency
        next_state[16] = predicted_makespan  # Update time_elapsed

        return next_state

    def long_horizon_reward_estimation(
        self,
        replay_buffer,
        current_state: np.ndarray,
        action: int,
        plan_count: int = 20,
        discount_factor: float = 0.99,
        target_makespan: float = 1.0,
        target_energy: float = 1.0,
        beta: float = 1.0
    ) -> float:
        """
        Estimate cumulative future reward over planning horizon.

        Matches interface from zero_sarbrl_v3.py.
        This is the key method that uses GraphPerf-RT for model-based planning.
        """
        cumulative_reward = 0.0
        state = current_state.copy()

        with torch.no_grad():
            for i in range(plan_count):
                # Predict next state using GraphPerf-RT
                predicted_makespan = self.predict(state, action)

                # Calculate reward (matching zero_samfrl.py reward function)
                epsilon = 1e-3
                if predicted_makespan <= epsilon:
                    predicted_makespan = target_makespan

                makespan_reward = target_makespan / (predicted_makespan + epsilon)
                reward = beta * makespan_reward

                # Accumulate discounted reward
                cumulative_reward += (discount_factor ** i) * reward

                # Update state for next iteration
                state = self.predict_next_state(state, action)

        return cumulative_reward

    def get_stats(self) -> Dict:
        """Get prediction statistics."""
        return {
            'prediction_count': self.prediction_count,
            'mean_error': np.mean(self.prediction_errors) if self.prediction_errors else 0,
            'std_error': np.std(self.prediction_errors) if self.prediction_errors else 0,
        }


# =============================================================================
# COMPARISON EXPERIMENT
# =============================================================================
class WorldModelComparison:
    """
    Compares GraphPerf-RT world model vs FCN world model vs model-free.

    Based on zero_combined_twobench.py structure.
    """

    def __init__(
        self,
        graphperf_model_path: str,
        graphperf_config_path: str,
        output_dir: Path
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GraphPerf-RT world model
        if graphperf_model_path and graphperf_config_path:
            try:
                self.graphperf_model = GraphPerfWorldModel(
                    model_path=graphperf_model_path,
                    config_path=graphperf_config_path
                )
            except Exception as e:
                logging.warning(f"Could not load GraphPerf-RT model: {e}")
                self.graphperf_model = None
        else:
            self.graphperf_model = None

        # Results storage
        self.results = {
            'MAMBRL_FCN': [],
            'MAMBRL_GraphPerf': [],
            'MAMFRL': [],
        }

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"world_model_comparison_{timestamp}.log"

    def log(self, message: str):
        """Write to log file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        logging.info(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def evaluate_prediction_accuracy(
        self,
        test_data: pd.DataFrame,
        world_model: Optional[GraphPerfWorldModel] = None
    ) -> Dict:
        """
        Evaluate world model prediction accuracy on test data.

        Returns metrics: RMSE, MAPE, R2
        """
        if world_model is None:
            return {'rmse': float('inf'), 'mape': float('inf'), 'r2': 0}

        predictions = []
        actuals = []

        for _, row in test_data.iterrows():
            # Construct state from row
            state = np.array([
                row.get('freq', 0),
                row.get('cores', 0),
                row.get('utilization', 0),
                # ... add more state components as needed
            ], dtype=np.float32)

            action = int(row.get('freq', 0))

            pred_makespan = world_model.predict(state, action)
            actual_makespan = row.get('makespan', row.get('time_elapsed', 0))

            predictions.append(pred_makespan)
            actuals.append(actual_makespan)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((predictions - actuals) / (actuals + 1e-8))) * 100

        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'n_samples': len(predictions)
        }

    def run_comparison(
        self,
        benchmarks: List[Dict],
        seeds: List[int] = [42, 123, 456],
        epochs: int = 200
    ) -> Dict:
        """
        Run full comparison experiment.

        Pattern: Based on zero_combined_twobench.py multi-seed structure.
        """
        self.log("="*70)
        self.log("WORLD MODEL COMPARISON EXPERIMENT")
        self.log(f"Benchmarks: {[b['benchmark'] for b in benchmarks]}")
        self.log(f"Seeds: {seeds}")
        self.log(f"Epochs: {epochs}")
        self.log("="*70)

        all_results = []

        for seed in seeds:
            self.log(f"\n{'='*60}")
            self.log(f"SEED: {seed}")
            self.log(f"{'='*60}")

            # Set random seed
            random.seed(seed)
            np.random.seed(seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(seed)

            # Run each method
            for method in ['MAMFRL', 'MAMBRL_FCN', 'MAMBRL_GraphPerf']:
                self.log(f"\n--- Method: {method} (seed={seed}) ---")

                result = {
                    'method': method,
                    'seed': seed,
                    'epochs': epochs,
                    'convergence_epoch': None,
                    'final_makespan': None,
                    'final_energy': None,
                    'training_time': None,
                }

                # Simulate or load results
                # In production, this would run actual training
                result['status'] = 'placeholder'

                all_results.append(result)
                self.results[method].append(result)

        # Generate comparison outputs
        self.generate_comparison_report(all_results)

        return all_results

    def generate_comparison_report(self, results: List[Dict]):
        """Generate comparison report and plots."""
        report_file = self.output_dir / 'comparison_report.txt'

        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("WORLD MODEL COMPARISON REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*70 + "\n\n")

            # Aggregate by method
            method_stats = {}
            for r in results:
                method = r['method']
                if method not in method_stats:
                    method_stats[method] = []
                method_stats[method].append(r)

            for method, runs in method_stats.items():
                f.write(f"\n{method}:\n")
                f.write(f"  Runs: {len(runs)}\n")
                # Add more statistics when real results available

        self.log(f"Report saved to: {report_file}")

        # Generate LaTeX table
        self.generate_latex_table(results)

    def generate_latex_table(self, results: List[Dict]):
        """Generate LaTeX comparison table for paper."""
        table_file = self.output_dir / 'world_model_comparison_table.tex'

        with open(table_file, 'w') as f:
            f.write(r"\begin{table}[t]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{World Model Comparison: GraphPerf-RT vs FCN Environment Model}" + "\n")
            f.write(r"\label{tab:world_model}" + "\n")
            f.write(r"\begin{tabular}{lcccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Method & Conv. Epoch & Makespan (s) & Energy (J) & Planning Time (ms) \\" + "\n")
            f.write(r"\midrule" + "\n")
            f.write(r"MAMFRL (Model-Free) & -- & -- & -- & 0 \\" + "\n")
            f.write(r"MAMBRL + FCN & -- & -- & -- & -- \\" + "\n")
            f.write(r"\textbf{MAMBRL + GraphPerf-RT} & -- & -- & -- & -- \\" + "\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")

        self.log(f"LaTeX table saved to: {table_file}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Compare GraphPerf-RT world model vs FCN environment model'
    )
    parser.add_argument(
        '--model', type=str,
        default='../GAT_EFG-new_training/checkpoints/best_model.pt',
        help='Path to GraphPerf-RT model checkpoint'
    )
    parser.add_argument(
        '--config', type=str,
        default='../GAT_EFG-new_training/configs/model_config_v2.yaml',
        help='Path to GraphPerf-RT config'
    )
    parser.add_argument(
        '--epochs', type=int,
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+',
        default=[42, 123, 456],
        help='Random seeds for reproducibility'
    )
    parser.add_argument(
        '--output', type=str,
        default='world_model_results',
        help='Output directory'
    )
    parser.add_argument(
        '--benchmarks', type=str, nargs='+',
        default=['fft', 'strassen', 'fib'],
        help='Benchmarks to run'
    )

    args = parser.parse_args()

    # Build benchmark configs
    benchmarks = [
        {"benchmark": b, "variant": "bin-omp-tasks", "input_arg": "default"}
        for b in args.benchmarks
    ]

    # Run comparison
    comparison = WorldModelComparison(
        graphperf_model_path=args.model,
        graphperf_config_path=args.config,
        output_dir=Path(args.output)
    )

    results = comparison.run_comparison(
        benchmarks=benchmarks,
        seeds=args.seeds,
        epochs=args.epochs
    )

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
