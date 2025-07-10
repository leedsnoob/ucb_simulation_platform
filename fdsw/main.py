import argparse
import yaml
from tqdm import tqdm
import sys
import os
import pandas as pd
import numpy as np
import multiprocessing
import itertools
import threading
import traceback

# Ensure the project root is in the Python path for module resolution
# This points to the workspace root, two levels up from this script's location.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from project_root.data.movielens_handler import MovieLensDataHandler
from project_root.data.obd_handler import OBDDataHandler
from project_root.drift.scene_generator import create_drift_scenario
# Import base algorithms
from project_root.bandits.ucb import UCB1
from project_root.bandits.ts import ThompsonSampling
from project_root.bandits.lints import LinTS
from project_root.bandits.ucb_fdsw import UCB_SW
from project_root.bandits.ucb_dsw import UCB_D, UCB_DSW
from project_root.bandits.etc import ETC

def get_algorithm_by_name(name: str, params: dict, n_arms: int, n_dims: int):
    """Factory function to create a single algorithm instance by name."""
    if name == "UCB1":
        return UCB1(n_arms=n_arms, **params)
    elif name == "UCB_SW":
        return UCB_SW(n_arms=n_arms, **params)
    elif name == "UCB_D":
        return UCB_D(n_arms=n_arms, **params)
    elif name == "UCB_DSW":
        return UCB_DSW(n_arms=n_arms, **params)
    elif name == "ETC":
        return ETC(n_arms=n_arms, **params)
    elif name == "ThompsonSampling":
        return ThompsonSampling(n_arms=n_arms, **params)
    elif name == "LinTS":
        return LinTS(d=n_dims, **params)
    else:
        raise ValueError(f"Unknown base algorithm: {name}")

def _update_progress(queue, total_tasks: int):
    """Listens on a queue and updates a tqdm progress bar."""
    with tqdm(total=total_tasks, desc="Overall Experiment Progress") as pbar:
        for _ in range(total_tasks):
            queue.get()
            pbar.update(1)

def run_base_experiment(config: dict, run_id: int, algo_config: dict, data_bundle: dict, progress_queue=None, is_parallel: bool = False):
    """
    Runs a single, complete experiment for one base algorithm.
    """
    algo_name = algo_config['name']
    algo_params = algo_config['params']
    
    try:
        process_id = os.getpid()
        log_message = f"[PID:{process_id}] Running Task: Run #{run_id+1}, Algo: {algo_name.upper()}"
        if not is_parallel:
            print(log_message)

        # 1. Unpack pre-loaded data
        data = data_bundle['data']
        num_timesteps = data_bundle['num_timesteps']
        n_arms_total = data_bundle['n_arms_total']
        n_dims = data_bundle['n_dims']

        # 2. Initialize Algorithm
        algorithm = get_algorithm_by_name(algo_name, algo_params, n_arms_total, n_dims)

        # 3. Run simulation
        rewards = [] # Changed from instantaneous_regrets

        for t in range(num_timesteps):
            row = data.iloc[t]
            
            chosen_arm_id = algorithm.select_arm(context=None)
            actual_reward = 0.0
            if chosen_arm_id == row['chosen_arm']:
                actual_reward = row['reward']
            
            algorithm.update(chosen_arm_id, actual_reward)
            
            rewards.append(actual_reward) # Changed from regret

        # 4. Save results
        results_df = pd.DataFrame({'reward': rewards}) # Changed column name
        
        filename = f"{config['experiment_name']}_{algo_name}_run_{run_id}.csv"
        results_dir = config['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(os.path.join(results_dir, filename), index=False)
        
        return filename
    finally:
        if progress_queue:
            progress_queue.put(1)

def worker_run_experiment(config, run_id, algo_config, data_bundle, progress_queue):
    """Helper function for multiprocessing."""
    try:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        return run_base_experiment(config, run_id, algo_config, data_bundle, progress_queue, is_parallel=True)
    except Exception as e:
        return f"ERROR in worker for args {config['experiment_name'], run_id, algo_config['name']}:\n{traceback.format_exc()}"

def main(config_path: str, num_workers_override: int = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully:")
    print(yaml.dump(config, indent=2))

    if 'algorithms' not in config:
        raise ValueError("This script requires the config to have an 'algorithms' list for base algorithm comparison.")

    # --- Stage 1: Centralized Data Loading ---
    print("\n--- STAGE 1: Loading and Preprocessing Data (ONCE) ---")
    if config['dataset'] == 'movielens':
        handler = MovieLensDataHandler()
    else:
        handler = OBDDataHandler(behavior_policy='bts')
    
    full_data = create_drift_scenario(handler, config['drift_type'])
    
    # Allow for a small subset for quick debugging
    subset_size = config.get('debug_subset_size', len(full_data))
    data = full_data.head(subset_size)
    print(f"\nUsing a subset of {subset_size} rows for this experiment run.")
    
    num_timesteps = len(data)
    n_arms_total = full_data['chosen_arm'].nunique()
    n_dims = full_data['context_vector'].iloc[0].shape[0] if 'context_vector' in full_data else 0

    data_bundle = {
        "data": data,
        "num_timesteps": num_timesteps,
        "n_arms_total": n_arms_total,
        "n_dims": n_dims
    }
    print("--- STAGE 1 COMPLETE ---")

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    tasks = list(itertools.product(
        [config],
        range(config['num_runs']),
        config['algorithms'],
        [data_bundle],
        [progress_queue]
    ))
    
    num_tasks = len(tasks)

    progress_thread = threading.Thread(
        target=_update_progress, args=(progress_queue, num_tasks), daemon=True
    )
    progress_thread.start()

    print(f"\n--- STAGE 2: Parallel Experiment Execution ---")
    print(f"Total tasks to run: {num_tasks} ({config['num_runs']} runs x {len(config['algorithms'])} algorithms)")

    try:
        num_workers = num_workers_override if num_workers_override else min(16, multiprocessing.cpu_count())
        print(f"Starting experiments in parallel on {num_workers} worker(s)...")
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(worker_run_experiment, tasks)

        # No need to join progress_thread if it's a daemon, but it's good practice to ensure queue is flushed.
        # Manager handles shutdown.
        
        print("\nAll experiments completed.")
        
        success_files = [r for r in results if not (isinstance(r, str) and r.startswith("ERROR"))]
        error_logs = [r for r in results if isinstance(r, str) and r.startswith("ERROR")]
        
        if success_files:
            print(f"Generated {len(success_files)} result file(s).")
        if error_logs:
            print(f"\n--- {len(error_logs)} TASK(S) FAILED WITH ERRORS ---")
            for error_log in error_logs:
                print("="*40)
                print(error_log)
                print("="*40)

    except Exception as e:
        print(f"\nAn error occurred during parallel execution: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run base bandit algorithm comparison experiments.")
    parser.add_argument(
        '--config', type=str, required=True, help="Path to the experiment configuration YAML file."
    )
    parser.add_argument(
        '--workers', type=int, default=None, help="Number of parallel worker processes."
    )
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"EXECUTING EXPERIMENT: {args.config}")
    print(f"{'='*80}\n")
    
    try:
        main(args.config, num_workers_override=args.workers)
        print(f"\n--- SUCCESS: Finished experiment for {args.config} ---")
    except Exception as e:
        print(f"\n--- ERROR: Experiment for {args.config} failed: {e} ---")
        traceback.print_exc()
    
    print(f"\n{'='*80}\n")
    print("Experiment run processed.") 