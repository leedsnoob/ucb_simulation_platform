import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import copy

# Add project root to the Python path to allow absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) # notebooks is one level down
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ucb_simulation_platform.data.obd_sim_handler import OBDSimHandler
from ucb_simulation_platform.environments.simulation_env import NonStationaryEnvironment

# --- Configuration ---
HORIZON = 5000
DRIFT_START_POINT = 2000
DRIFT_DURATION = 1000
FIGURES_DIR = '../figures'

def generate_summary_table(env_before: NonStationaryEnvironment, env_after: NonStationaryEnvironment):
    """Generates and prints a summary table of mu values before and after the drift."""
    print("\n--- OBD Gradual Drift µ Values Summary ---")
    
    # We use the arms that were set to drift initially
    arm_a, arm_b = env_after.drift_arm_pairs[0]
    
    summary_data = {
        "Arm ID": [f"Arm {arm_a}", f"Arm {arm_b}"],
        "µ Before Drift": [f"{env_before.true_mus[arm_a]:.4f}", f"{env_before.true_mus[arm_b]:.4f}"],
        "µ After Drift": [f"{env_after.true_mus[arm_a]:.4f}", f"{env_after.true_mus[arm_b]:.4f}"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_markdown(index=False))

def verify_obd_gradual_drift():
    """
    Runs a simulation on OBD to verify and visualize the gradual drift mechanism.
    """
    print("--- Verifying Gradual Drift Mechanism for OBD ---")
    
    # 1. Initialize Handler and Environment
    handler = OBDSimHandler(data_dir='obd/random')
    env = NonStationaryEnvironment(data_handler=handler)
    
    # Define which arms will drift by initializing the gradual drift
    env.init_gradual_drift(DRIFT_START_POINT, DRIFT_DURATION)
    
    # Store initial arm IDs for consistent labeling
    drifting_arm_ids_map = {
        'Best 1': env.best_arm_1, 'Best 2': env.best_arm_2,
        'Worst 1': env.worst_arm_1, 'Worst 2': env.worst_arm_2
    }
    
    mus_history = []

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

    # Plot all other arms as a faint background
    all_drifting_ids = list(drifting_arm_ids_map.values())
    for arm_id in range(env.n_arms):
        if arm_id not in all_drifting_ids:
            ax.plot(mus_df.index, mus_df[arm_id], color='grey', alpha=0.2, linewidth=0.5)

    # Highlight the drifting arms
    line_styles = {'Best 1': '-', 'Best 2': '--', 'Worst 1': ':', 'Worst 2': '-.'}
    colors = {'Best 1': 'blue', 'Best 2': 'cyan', 'Worst 1': 'red', 'Worst 2': 'magenta'}

    for label, arm_id in drifting_arm_ids_map.items():
        ax.plot(mus_df.index, mus_df[arm_id], label=f'Arm {arm_id} ({label})', 
                linestyle=line_styles[label], linewidth=3, color=colors[label])

    drift_end_point = DRIFT_START_POINT + DRIFT_DURATION
    ax.axvspan(DRIFT_START_POINT, drift_end_point, color='orange', alpha=0.3, label='Gradual Drift Window')
    
    ax.set_title('OBD Gradual Drift Verification: Top-2/Bottom-2 Arm Swap', fontsize=18)
    ax.set_xlabel('Time Step (t)', fontsize=12)
    ax.set_ylabel('True Mean Reward (µ)', fontsize=12)
    ax.legend(title='Key Arms', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    
    script_dir = os.path.dirname(__file__)
    save_path = os.path.abspath(os.path.join(script_dir, FIGURES_DIR, 'obd_gradual_drift_verification_plot.png'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    print(f"Gradual drift verification plot saved to: {save_path}")

if __name__ == '__main__':
    verify_obd_gradual_drift() 