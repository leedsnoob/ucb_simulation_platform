import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# --- Configuration ---
RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
EXPERIMENT_NAME = 'MovieLens-Abrupt-Paper-Params-QUICK-Test'
ALGORITHMS = ['UCB1', 'UCB_DSW']
NUM_RUNS = 1 # We ran it once

# --- Create Figure Directory ---
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# --- Data Loading and Processing ---
all_dfs = []
for algo in ALGORITHMS:
    for run_id in range(NUM_RUNS):
        filename = f"{EXPERIMENT_NAME}_{algo}_run_{run_id}.csv"
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['algorithm_name'] = algo
            df['run_id'] = run_id
            df['timestep'] = np.arange(len(df))
            df['cumulative_regret'] = df['instantaneous_regret'].cumsum()
            all_dfs.append(df)
        else:
            print(f"Warning: Result file not found at {path}")

if not all_dfs:
    print("Error: No data loaded. Exiting.")
    exit()

full_df = pd.concat(all_dfs, ignore_index=True)

# --- Plotting ---
print("Plotting results...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 10))

colors = plt.cm.get_cmap('viridis', len(ALGORITHMS))

for i, algo_name in enumerate(ALGORITHMS):
    # For this simple script, we don't calculate std dev as there's only one run
    algo_df = full_df[full_df['algorithm_name'] == algo_name]
    if not algo_df.empty:
        ax.plot(algo_df['timestep'], algo_df['cumulative_regret'], label=algo_name, color=colors(i), linewidth=2)

# --- Annotations ---
total_timesteps = full_df['timestep'].max()
if total_timesteps > 0:
    t1 = total_timesteps // 3
    t2 = 2 * total_timesteps // 3
    ax.axvline(x=t1, color='grey', linestyle='--', label=f'Drift Point 1 (t={t1})')
    ax.axvline(x=t2, color='grey', linestyle='--', label=f'Drift Point 2 (t={t2})')

ax.set_title(f"Cumulative Regret for {EXPERIMENT_NAME}", fontsize=18)
ax.set_xlabel("Timesteps", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.legend()
fig.tight_layout()

# --- Save Figure ---
fig_path = os.path.join(FIGURES_DIR, f"{EXPERIMENT_NAME}_simple_plot.png")
plt.savefig(fig_path)

print(f"Plot successfully saved to: {fig_path}")

# --- Summary Table ---
print("\n--- Final Regret Summary ---")
final_regrets = full_df.groupby('algorithm_name')['cumulative_regret'].max().reset_index()
print(final_regrets.to_markdown(index=False)) 