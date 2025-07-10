import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import argparse

# Add project root to the Python path to allow absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) # notebooks is one level down
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ucb_simulation_platform.data.movielens_sim_handler import MovieLensSimHandler
from ucb_simulation_platform.data.obd_sim_handler import OBDSimHandler
from environments.simulation_env import StationaryEnvironment

# --- Configuration ---
FIGURES_DIR = '../figures'

def analyze_reward_distributions(dataset: str):
    """
    Loads the specified simulation handler and plots the reward distributions
    of the initial best and worst arms to visualize their difference.
    """
    print(f"--- Analyzing Reward Distributions for {dataset.upper()} ---")
    
    if dataset == 'movielens':
        handler = MovieLensSimHandler(n_clusters=9, data_dir='ml-1m')
        env = StationaryEnvironment(data_handler=handler)
        bins = np.arange(1, 7) - 0.5
        x_ticks = [1, 2, 3, 4, 5]
        x_label = 'Rating (Reward)'
        title = 'Reward Distribution: Best vs. Worst Arm (MovieLens User Clusters)'
        save_name = 'movielens_reward_distribution_plot.png'
    elif dataset == 'obd':
        handler = OBDSimHandler(data_dir='obd/random')
        env = StationaryEnvironment(data_handler=handler)
        bins = np.arange(0, 3) - 0.5
        x_ticks = [0, 1]
        x_label = 'Click (Reward)'
        title = 'Reward Distribution: Best vs. Worst Arm (OBD items)'
        save_name = 'obd_reward_distribution_plot.png'
    else:
        raise ValueError("Invalid dataset specified. Choose 'movielens' or 'obd'.")

    sorted_arms = sorted(env.get_true_means().items(), key=lambda item: item[1])
    worst_arm_id = sorted_arms[0][0]
    best_arm_id = sorted_arms[-1][0]
    
    worst_arm_pool = handler.reward_pools[worst_arm_id]
    best_arm_pool = handler.reward_pools[best_arm_id]
    
    print(f"Initial Best Arm: {best_arm_id} (µ = {env.true_mus[best_arm_id]:.4f})")
    print(f"Initial Worst Arm: {worst_arm_id} (µ = {env.true_mus[worst_arm_id]:.4f})")
    
    # Plotting
    print("\nPlotting reward distributions...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.hist(best_arm_pool, bins=bins, density=True, alpha=0.7, label=f'Best Arm ({best_arm_id}) Rewards', color='green')
    ax.hist(worst_arm_pool, bins=bins, density=True, alpha=0.7, label=f'Worst Arm ({worst_arm_id}) Rewards', color='red')
    
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_xticks(x_ticks)
    ax.legend()
    fig.tight_layout()
    
    # Save the figure
    script_dir = os.path.dirname(__file__)
    save_path = os.path.abspath(os.path.join(script_dir, FIGURES_DIR, save_name))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    print(f"Reward distribution plot saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and plot reward distributions for a dataset.")
    parser.add_argument(
        '--dataset', type=str, required=True, choices=['movielens', 'obd'],
        help="The dataset to analyze ('movielens' or 'obd')."
    )
    args = parser.parse_args()
    analyze_reward_distributions(args.dataset) 