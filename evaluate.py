import sys
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to the Python path to allow absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ucb_simulation_platform.data.movielens_sim_handler import MovieLensSimHandler
from ucb_simulation_platform.data.obd_sim_handler import OBDSimHandler
from ucb_simulation_platform.bandits.ucb import UCB1
from ucb_simulation_platform.bandits.ts import ThompsonSampling
from ucb_simulation_platform.bandits.ucb_d import UCB_D

def load_results(results_dir: str, experiment_name: str) -> pd.DataFrame:
    """Loads all CSV files for a given experiment into a single DataFrame."""
    search_path = os.path.join(results_dir, f"{experiment_name}_*.csv")
    file_list = glob.glob(search_path)
    if not file_list:
        raise FileNotFoundError(f"No result files found for pattern: {search_path}")

    df_list = [pd.read_csv(f) for f in file_list]
    return pd.concat(df_list, ignore_index=True)

def plot_cumulative_regret(df: pd.DataFrame, config: dict, figures_dir: str):
    """Plots the mean cumulative regret with a confidence interval and drift annotations."""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # --- Calculate and Plot Cumulative Regret ---
    for algo_name in df['algorithm_name'].unique():
        algo_df = df[df['algorithm_name'] == algo_name]
        # Group by timestep and calculate mean and std of cumulative regret
        agg_df = algo_df.groupby('timestep')['cumulative_regret'].agg(['mean', 'std'])
        
        plt.plot(agg_df.index, agg_df['mean'], label=algo_name)
        plt.fill_between(
            agg_df.index,
            agg_df['mean'] - agg_df['std'],
            agg_df['mean'] + agg_df['std'],
            alpha=0.2
        )

    # --- Annotate Drifts ---
    if 'drift_config' in config and config['drift_config']:
        for event in config['drift_config']:
            position = event.get('position') # For abrupt drifts
            start = event.get('start')       # For gradual drifts
            duration = event.get('duration') # For gradual drifts

            # Handle abrupt drift
            if position:
                # Only draw the line if it's within the plot's horizon
                if position <= ax.get_xlim()[1]:
                    ax.axvline(x=position, color='r', linestyle='--', linewidth=2, label=f'Abrupt Drift at t={position}')
            
            # Handle gradual drift
            elif start is not None and duration is not None:
                # Only draw the region if it's at least partially visible
                if start <= ax.get_xlim()[1]:
                    end = start + duration
                    ax.axvspan(start, end, color='orange', alpha=0.3, label=f'Gradual Drift ({start}-{end})')
    
    # To avoid duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title(f"Cumulative Regret: {config['experiment_name']}")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Regret")
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"{config['experiment_name']}_cumulative_regret.png"))
    plt.close()

def plot_rolling_reward(df: pd.DataFrame, config: dict, figures_dir: str, window_size=500):
    """Plots the rolling mean reward for each algorithm."""
    print(f"Plotting Rolling Mean Reward (window={window_size}) for {config['experiment_name']}...")
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    df['rolling_reward'] = df.groupby(['algorithm_name', 'run_id'])['reward'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    
    agg_df = df.groupby(['algorithm_name', 'timestep'])['rolling_reward'].agg('mean').reset_index()

    for algo_name in agg_df['algorithm_name'].unique():
        algo_df = agg_df[agg_df['algorithm_name'] == algo_name]
        ax.plot(algo_df['timestep'], algo_df['rolling_reward'], label=algo_name)

    # --- Annotate Drifts ---
    if 'drift_config' in config and config['drift_config']:
        for event in config['drift_config']:
            position = event.get('position') # For abrupt drifts
            start = event.get('start')       # For gradual drifts
            duration = event.get('duration') # For gradual drifts

            # Handle abrupt drift
            if position:
                if position <= ax.get_xlim()[1]:
                    ax.axvline(x=position, color='r', linestyle='--', linewidth=2, label=f'Abrupt Drift at t={position}')
            
            # Handle gradual drift
            elif start is not None and duration is not None:
                if start <= ax.get_xlim()[1]:
                    end = start + duration
                    ax.axvspan(start, end, color='orange', alpha=0.3, label=f'Gradual Drift ({start}-{end})')

    # To avoid duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title(f"Rolling Mean Reward (window={window_size}): {config['experiment_name']}")
    plt.xlabel("Timestep")
    plt.ylabel("Rolling Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, f"{config['experiment_name']}_rolling_reward.png"))
    plt.close()
    print(f"  - Saved to figures/{config['experiment_name']}_rolling_reward.png")


def generate_summary_table(df: pd.DataFrame):
    """Generates and prints a summary table of the final cumulative regret."""
    print("\n--- Final Performance Summary ---")
    
    # Find the final cumulative regret for each run
    final_regrets = df.groupby(['algorithm_name', 'run_id'])['cumulative_regret'].max().reset_index()
    
    # Calculate mean and std. If std doesn't exist (e.g., num_runs=1), it will be NaN.
    summary_stats = final_regrets.groupby('algorithm_name')['cumulative_regret'].agg(['mean', 'std']).reset_index()
    
    # Ensure 'std' column exists and fill NaN values with 0
    if 'std' not in summary_stats.columns:
        summary_stats['std'] = 0.0
    summary_stats['std'] = summary_stats['std'].fillna(0)
    
    summary_stats.rename(columns={'mean': 'Mean Final Regret', 'std': 'Std Dev of Final Regret'}, inplace=True)
    summary_stats = summary_stats.sort_values(by='Mean Final Regret').reset_index(drop=True)
    
    print(summary_stats.to_markdown(index=False))

def main(config_path: str):
    """Main function to load, plot, and summarize experiment results."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_name = config['experiment_name']
    results_dir = config.get('results_dir', 'results')
    figures_dir = 'figures' 
    os.makedirs(figures_dir, exist_ok=True)

    try:
        all_data = load_results(results_dir, exp_name)
        if all_data.empty:
            print("No results found to evaluate.")
            return

        # Calculate cumulative regret from the raw 'regret' column
        all_data['cumulative_regret'] = all_data.groupby(['algorithm_name', 'run_id'])['regret'].cumsum()
        
        # Generate plots and summary
        print(f"Plotting Cumulative Regret for {exp_name}...")
        plot_cumulative_regret(all_data, config, figures_dir)
        print(f"  - Saved to figures/{exp_name}_cumulative_regret.png")

        plot_rolling_reward(all_data.copy(), config, figures_dir)
        
        generate_summary_table(all_data)

    except FileNotFoundError as e:
        print(f"Error loading results: {e}")
    except KeyError as e:
        print(f"An unexpected error occurred: 'Column not found: {e}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate experiment results.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config) 