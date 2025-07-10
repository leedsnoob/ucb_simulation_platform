import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd

# Add project root to the Python path to allow absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) # notebooks is one level down
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ucb_simulation_platform.data.movielens_sim_handler import MovieLensSimHandler
from ucb_simulation_platform.environments.simulation_env import NonStationaryEnvironment

# --- Configuration ---
HORIZON = 30000
DRIFT_START_POINT = 10000
DRIFT_DURATION = 5000
N_CLUSTERS = 9
FIGURES_DIR = '../figures'

def verify_gradual_drift_mechanism():
    """
    Runs a simulation to verify and visualize the gradual drift mechanism.
    """
    print("--- Verifying Gradual Drift Mechanism for MovieLens ---")
    
    # 1. Initialize Handler and Environment
    handler = MovieLensSimHandler(n_clusters=N_CLUSTERS, data_dir='ml-1m')
    env = NonStationaryEnvironment(data_handler=handler)
    
    mus_history = []
    initial_best_arm_1 = env.best_arm_1
    initial_worst_arm_1 = env.worst_arm_1
    initial_best_arm_2 = env.best_arm_2
    initial_worst_arm_2 = env.worst_arm_2
    
    # UPDATED: No need to pass arm pairs anymore
    env.init_gradual_drift(DRIFT_START_POINT, DRIFT_DURATION)

    # 2. Run Simulation
    print(f"\nRunning simulation for {HORIZON} steps...")
    for t in range(HORIZON):
        env.step(arm_index=0, t=t)
        mus_history.append(env.get_true_means())

    # 3. Post-process and Plot
    print("\nProcessing results for plotting...")
    mus_df = pd.DataFrame(mus_history)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))

    # Highlight the drifting arms
    drifting_arm_ids = [initial_best_arm_1, initial_best_arm_2, initial_worst_arm_1, initial_worst_arm_2]
    line_styles = {'Best 1': '-', 'Best 2': '--', 'Worst 1': ':', 'Worst 2': '-.'}
    labels = {
        initial_best_arm_1: 'Best 1', initial_best_arm_2: 'Best 2',
        initial_worst_arm_1: 'Worst 1', initial_worst_arm_2: 'Worst 2'
    }

    for arm_id in drifting_arm_ids:
        label_key = labels[arm_id]
        ax.plot(mus_df.index, mus_df[arm_id], label=f'Arm {arm_id} ({label_key})', 
                linestyle=line_styles[label_key], linewidth=3)

    drift_end_point = DRIFT_START_POINT + DRIFT_DURATION
    ax.axvspan(DRIFT_START_POINT, drift_end_point, color='orange', alpha=0.3, label='Gradual Drift Window')
    
    ax.set_title('Verification of Gradual Swap Drift: True Mean Reward (µ) vs. Time', fontsize=18)
    ax.set_xlabel('Time Step (t)', fontsize=12)
    ax.set_ylabel('True Mean Reward (µ)', fontsize=12)
    ax.legend(title='Drifting Arms')
    fig.tight_layout()
    
    script_dir = os.path.dirname(__file__)
    save_path = os.path.abspath(os.path.join(script_dir, FIGURES_DIR, 'gradual_drift_verification_plot.png'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Gradual drift verification plot saved to: {save_path}")


if __name__ == '__main__':
    verify_gradual_drift_mechanism() 