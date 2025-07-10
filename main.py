import sys
import os
import argparse
import yaml
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import importlib
import logging
import itertools
import traceback
import numpy as np

# Add project root to the Python path to allow absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import environment and data handlers
from ucb_simulation_platform.environments.simulation_env import NonStationaryEnvironment, StationaryEnvironment
from ucb_simulation_platform.data.movielens_sim_handler import MovieLensSimHandler
from ucb_simulation_platform.data.obd_sim_handler import OBDSimHandler
from ucb_simulation_platform.data.base_sim_handler import BaseSimHandler

# Import all bandit algorithms
from ucb_simulation_platform.bandits.ucb import UCB1
from ucb_simulation_platform.bandits.ts import ThompsonSampling
from ucb_simulation_platform.bandits.ucb_d import UCB_D
from ucb_simulation_platform.bandits.ucb_sw import UCB_SW
from ucb_simulation_platform.bandits.ucb_fdsw import FDSW_UCB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A mapping from algorithm name string (in YAML) to the actual class
ALGO_MAP = {
    "UCB1": UCB1,
    "ThompsonSampling": ThompsonSampling,
    "UCB_D": UCB_D,
    "UCB_SW": UCB_SW,
    "FDSW_UCB": FDSW_UCB
}

def get_algorithm_by_name(name: str, params: dict, n_arms: int):
    """Factory function to create an algorithm instance from config."""
    # The 'class' key allows us to use different names in the YAML for the same class
    class_name_str = params.pop("class", name) 
    
    if class_name_str in ALGO_MAP:
        AlgoClass = ALGO_MAP[class_name_str]
        # Pass the algorithm's descriptive name for logging purposes
        params['name'] = name 
        return AlgoClass(n_arms=n_arms, **params)
    else:
        raise ValueError(f"Unknown algorithm class: {class_name_str}")

def run_single_experiment(config, run_id, algo_config, data_bundle, seed):
    """
    Runs a single, complete experiment for one algorithm config and one run.
    Accepts unpacked arguments directly from pool.starmap.
    """
    algo_name = algo_config['name']
    algo_params = algo_config.get('params', {})
    
    try:
        # data = data_bundle['data'] # This is not needed for simulation
        num_timesteps = data_bundle['num_timesteps']
        n_arms = data_bundle['n_arms']
        handler = data_bundle['handler']
        # seed = data_bundle['seed'] # This line is removed as seed is now passed directly

        # Get the correct algorithm class
        if 'class' in algo_config:
            class_name = algo_config['class']
        else:
            class_name = algo_name

        # Pass horizon and drift_config for dynamic parameter calculation
        algo_params['horizon'] = num_timesteps
        algo_params['drift_config'] = config.get('drift_config', [])

        if class_name == "UCB1":
            from ucb_simulation_platform.bandits.ucb import UCB1
            algorithm = UCB1(n_arms=n_arms, **algo_params)
        elif class_name == "FDSW_UCB":
            from ucb_simulation_platform.bandits.ucb_fdsw import FDSW_UCB
            algorithm = FDSW_UCB(n_arms=n_arms, **algo_params)
        elif class_name == "UCB_D":
            from ucb_simulation_platform.bandits.ucb_d import UCB_D
            algorithm = UCB_D(n_arms=n_arms, **algo_params)
        elif class_name == "UCB_SW":
            from ucb_simulation_platform.bandits.ucb_sw import UCB_SW
            algorithm = UCB_SW(n_arms=n_arms, **algo_params)
        else:
             raise ValueError(f"Unknown algorithm class: {class_name}")

        # 3. Run simulation
        results = []
        
        # Environment gets the specific seed for this run
        env_class = StationaryEnvironment if config['drift_type'] == 'stationary' else NonStationaryEnvironment
        env = env_class(handler, seed=seed)
        if config['drift_type'] != 'stationary':
            env.init_drift(config.get('drift_config', []))

        for t in range(num_timesteps):
            chosen_arm = algorithm.select_arm()
            reward = env.step(chosen_arm, t)
            algorithm.update(chosen_arm, reward)
            
            regret = env.get_best_mu() - env.true_mus[chosen_arm]
            results.append({
                'timestep': t,
                'run_id': run_id,
                'algorithm_name': algo_name,
                'regret': regret,
                'reward': reward  # Add reward to the results
            })

        # DataFrame now includes timestep, run_id, algorithm_name, and regret
        results_df = pd.DataFrame(results)
        
        filename = f"{config['experiment_name']}_{algo_name}_run_{run_id}.csv"
        results_dir = config['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        # We save the raw results; cumulative regret will be calculated in evaluation
        results_df.to_csv(os.path.join(results_dir, filename), index=False)
        
        return filename
    except Exception as e:
        # Return a descriptive error string for debugging
        return f"ERROR in {algo_name} run {run_id}: {e}\n{traceback.format_exc()}"


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("--- Experiment Configuration ---")
    print(yaml.dump(config, indent=2))
    print("------------------------------")

    seed = config.get("seed", 42)
    handler_params = config.get("handler_params", {})
    
    # --- Stage 1: Centralized Data Loading ---
    print("Loading and preprocessing data...")
    if config['dataset'] == 'movielens':
        handler = MovieLensSimHandler(seed=seed, **handler_params)
    elif config['dataset'] == 'obd':
        handler = OBDSimHandler(seed=seed, **handler_params)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
        
    num_timesteps = config['horizon']
    n_arms = handler.n_arms

    # This was the missing piece. Now data_bundle contains the handler
    # which has access to reward pools, true means, etc.
    data_bundle = {
        "handler": handler,
        "num_timesteps": num_timesteps,
        "n_arms": n_arms
    }
    print("Data loaded successfully.")
    
    tasks = []
    for algo_config in config['algorithms']:
        for run_id in range(config['num_runs']):
            # Each run gets a unique, deterministic seed
            run_seed = seed + run_id
            tasks.append((config, run_id, algo_config, data_bundle, run_seed))

    # --- Run experiments in parallel ---
    print(f"Starting {len(tasks)} tasks in parallel using {cpu_count()} workers...")
    with Pool(processes=cpu_count()) as pool:
        # Use starmap to unpack the tuples in tasks for the worker function
        results = list(tqdm(pool.starmap(run_single_experiment, tasks), total=len(tasks)))

    print("\n--- Experiment Run Complete ---")
    print("Summary of results:")
    for res in results:
        print(f"  - {res}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run simulation experiments for UCB algorithms.")
    parser.add_argument(
        '--config', type=str, required=True, 
        help="Path to the experiment configuration YAML file (e.g., 'config/movielens_sim.yaml')."
    )
    args = parser.parse_args()
    main(args.config) 