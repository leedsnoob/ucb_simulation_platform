import matplotlib
matplotlib.use('Agg')
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# --- Constants for Path Management ---
# The script is in project_root/fdsw/, so we go up two levels to get to the workspace root.
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, 'results')
FIGURES_DIR = os.path.join(WORKSPACE_ROOT, 'figures')


def load_results(experiment_name: str) -> pd.DataFrame:
    """Loads all CSV results for a given experiment name into a single DataFrame."""
    print(f"Loading results for experiment: {experiment_name}")
    search_pattern = os.path.join(RESULTS_DIR, f"{experiment_name}_*.csv")
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        print(f"Warning: No files found for pattern: {search_pattern}")
        return pd.DataFrame()

    all_dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        filename = os.path.basename(path)
        match = re.match(f"{experiment_name}_(.*)_run_(\d+).csv", filename)
        if match:
            df['algorithm_name'] = match.group(1)
            df['run_id'] = int(match.group(2))
            df['timestep'] = np.arange(len(df))
            all_dfs.append(df)
    
    if not all_dfs:
        print("Warning: Files were found, but no data could be parsed from filenames.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def plot_results(df: pd.DataFrame, experiment_name: str):
    """Plots the cumulative reward curves with annotations for drift."""
    print("Plotting results...")
    # Calculate cumulative reward instead of regret
    df['cumulative_reward'] = df.groupby(['algorithm_name', 'run_id'])['reward'].cumsum()
    
    agg_df = df.groupby(['algorithm_name', 'timestep'])['cumulative_reward'].agg(['mean', 'std']).reset_index()
    agg_df.fillna(0, inplace=True) # Fill std for single runs with 0

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))

    algorithms = agg_df['algorithm_name'].unique()
    colors = plt.cm.get_cmap('viridis', len(algorithms))

    for i, algo_name in enumerate(algorithms):
        algo_df = agg_df[agg_df['algorithm_name'] == algo_name]
        ax.plot(algo_df['timestep'], algo_df['mean'], label=algo_name, color=colors(i), linewidth=2)
        ax.fill_between(
            algo_df['timestep'],
            algo_df['mean'] - algo_df['std'],
            algo_df['mean'] + algo_df['std'],
            color=colors(i),
            alpha=0.2
        )

    total_timesteps = df['timestep'].max()
    if total_timesteps > 0:
        t1 = total_timesteps // 3
        t2 = 2 * total_timesteps // 3
        ax.axvline(x=t1, color='grey', linestyle='--', label=f'Drift Point 1 (t={t1})')
        ax.axvline(x=t2, color='grey', linestyle='--', label=f'Drift Point 2 (t={t2})')

    ax.set_title(f"Cumulative Reward for {experiment_name}", fontsize=18)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.legend()
    fig.tight_layout()
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, f"{experiment_name}_reward_plot.png")
    plt.savefig(fig_path)
    print(f"Plot saved to {fig_path}")
    plt.close(fig) # Close the figure to free up memory


def generate_summary_table(df: pd.DataFrame):
    """Generates and prints a summary table of the final cumulative rewards."""
    print("\n--- Final Cumulative Reward Summary ---")
    
    if df.empty:
        print("No data to summarize.")
        return

    # Ensure cumulative_reward is calculated if not present (might be done in plot_results)
    if 'cumulative_reward' not in df.columns:
        df['cumulative_reward'] = df.groupby(['algorithm_name', 'run_id'])['reward'].cumsum()

    final_rewards = df.groupby(['algorithm_name', 'run_id'])['cumulative_reward'].max().reset_index()
    summary = final_rewards.groupby('algorithm_name')['cumulative_reward'].agg(['mean', 'std']).reset_index()
    # Sort by mean reward in descending order (higher is better)
    summary = summary.sort_values(by='mean', ascending=False).reset_index(drop=True)
    
    if not summary.empty:
        best_algo_idx = summary['mean'].idxmax()
        summary['mean'] = summary['mean'].apply(lambda x: f"{x:,.2f}")
        summary['std'] = summary['std'].fillna(0).apply(lambda x: f"{x:,.2f}")
        summary.at[best_algo_idx, 'algorithm_name'] = f"**{summary.at[best_algo_idx, 'algorithm_name']}**"
        print(summary.to_markdown(index=False))


def main(experiment_name: str):
    """Main function to load, plot, and summarize experiment results."""
    all_data = load_results(experiment_name)
    
    if all_data.empty:
        print("No result files found for the given experiment name. Exiting.")
        return

    plot_results(all_data, experiment_name)
    generate_summary_table(all_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate and visualize FDSW experiment results.")
    parser.add_argument('-e', '--experiment_name', type=str, required=True,
                        help="The base name of the experiment to evaluate.")
    
    args = parser.parse_args()
    main(args.experiment_name)
