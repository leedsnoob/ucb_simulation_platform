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
DRIFT_POINT = 2500
FIGURES_DIR = '../figures'

def generate_summary_table(env_before: NonStationaryEnvironment, env_after: NonStationaryEnvironment):
    """Generates and prints a summary table of mu values before and after the drift."""
    print("\n--- OBD µ Values Summary ---")
    
    arms = env_before.best_arm_1, env_before.best_arm_2, env_before.worst_arm_1, env_before.worst_arm_2
    arm_labels = ['Best 1', 'Best 2', 'Worst 1', 'Worst 2']
    
    summary_data = {
        "Arm Label": arm_labels,
        "Arm ID": [f"Arm {arm}" for arm in arms],
        "µ Before Drift": [f"{env_before.true_mus[arm]:.4f}" for arm in arms],
        "µ After Drift": [f"{env_after.true_mus[arm]:.4f}" for arm in arms]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_markdown(index=False))


def verify_obd_drift_mechanism():
    """
    Runs a short simulation on OBD to verify and visualize the drift mechanism.
    """
    print("--- Verifying Drift Mechanism for OBD ---")
    
    # 1. Initialize Handler and Environment
    handler = OBDSimHandler(data_dir='obd/random')
    env = NonStationaryEnvironment(data_handler=handler)
    
    env_before_drift = copy.deepcopy(env)
    
    mus_history = []
    initial_best_arm_1 = env.best_arm_1
    initial_worst_arm_1 = env.worst_arm_1
    initial_best_arm_2 = env.best_arm_2
    initial_worst_arm_2 = env.worst_arm_2
    
    print(f"\nInitial State:")
    print(f"  - Best Arms: {initial_best_arm_1} (µ={env.true_mus[initial_best_arm_1]:.4f}), {initial_best_arm_2} (µ={env.true_mus[initial_best_arm_2]:.4f})")
    print(f"  - Worst Arms: {initial_worst_arm_1} (µ={env.true_mus[initial_worst_arm_1]:.4f}), {initial_worst_arm_2} (µ={env.true_mus[initial_worst_arm_2]:.4f})")

    # 2. Run Simulation
    print(f"\nRunning simulation for {HORIZON} steps...")
    for t in range(HORIZON):
        mus_history.append(env.get_true_means().copy()) # Use copy to store snapshot
        
        if t == DRIFT_POINT:
            print(f"\n>>> DRIFT EVENT at t={t} <<<")
            env.drift_swap_extremes()

    # 3. Post-process and Plot
    print("\nProcessing results for plotting...")
    mus_df = pd.DataFrame(mus_history)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot all other arms as a faint background
    for arm_id in range(env.n_arms):
        if arm_id not in [initial_best_arm_1, initial_best_arm_2, initial_worst_arm_1, initial_worst_arm_2]:
            ax.plot(mus_df.index, mus_df[arm_id], color='grey', alpha=0.2, linewidth=0.5)

    # Highlight the arms that are supposed to drift
    extreme_arms = {
        f'Arm {initial_best_arm_1} (Best 1)': (initial_best_arm_1, '-', 'blue'),
        f'Arm {initial_best_arm_2} (Best 2)': (initial_best_arm_2, '--', 'cyan'),
        f'Arm {initial_worst_arm_1} (Worst 1)': (initial_worst_arm_1, ':', 'red'),
        f'Arm {initial_worst_arm_2} (Worst 2)': (initial_worst_arm_2, '-.', 'magenta')
    }

    for label, (arm_id, line_style, color) in extreme_arms.items():
        ax.plot(mus_df.index, mus_df[arm_id], label=label, linestyle=line_style, linewidth=3, color=color)

    # To perfectly align the line with the step change which happens *between* DRIFT_POINT and DRIFT_POINT+1,
    # we draw the line at DRIFT_POINT + 0.5.
    ax.axvline(x=DRIFT_POINT + 0.5, color='black', linestyle='-', lw=2, label=f'Drift Event at t={DRIFT_POINT}')
    
    ax.set_title('OBD Drift Verification: Top-2/Bottom-2 Arm Swap', fontsize=18)
    ax.set_xlabel('Time Step (t)', fontsize=12)
    ax.set_ylabel('True Mean Reward (µ)', fontsize=12)
    ax.legend(title='Key Arms', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    
    script_dir = os.path.dirname(__file__)
    save_path = os.path.abspath(os.path.join(script_dir, FIGURES_DIR, 'obd_drift_verification_plot.png'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    print(f"Drift verification plot saved to: {save_path}")

    # Generate the summary table after plotting
    generate_summary_table(env_before_drift, env)


if __name__ == '__main__':
    verify_obd_drift_mechanism() 