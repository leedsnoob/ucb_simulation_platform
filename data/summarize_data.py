import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.movielens_handler import MovieLensDataHandler
from data.obd_handler import OBDDataHandler
from drift.scene_generator import create_drift_scenario, get_drift_points, get_drift_window

# --- Plotting Functions ---

def plot_reward_mean_with_drift(data, window_size, title, save_path, abrupt_drift_points=None, gradual_drift_window=None):
    """
    Plots a rolling reward mean and visualizes drift points and windows.
    """
    print(f"  Plotting: Rolling Reward Mean...")
    plt.figure(figsize=(14, 7))
    
    data['reward'].rolling(window=window_size, min_periods=1).mean().plot(label=f'Rolling Mean (window={window_size})')
    
    if abrupt_drift_points:
        for point, label in abrupt_drift_points.items():
            plt.axvline(x=point, color='r', linestyle='--', label=label)
            
    if gradual_drift_window:
        plt.axvspan(gradual_drift_window[0], gradual_drift_window[1], color='orange', alpha=0.3, label='Gradual Drift Window')

    plt.title(title, fontsize=16)
    plt.xlabel("Interaction Index", fontsize=12)
    plt.ylabel("Average Reward / CTR", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"    ...saved to {save_path}")

def plot_top_arm_ctr_stages(data_before, data_after, top_n, title, save_path, t1, t2):
    print(f"  Plotting: Top-{top_n} Arm CTR across Stages...")
    top_arms = data_before['chosen_arm'].value_counts().nlargest(top_n).index
    
    stages_data = {
        'Before Drift (t < t1)': data_after[data_after.index < t1],
        'During Drift (t1 <= t < t2)': data_after[(data_after.index >= t1) & (data_after.index < t2)],
        'After Drift (t >= t2)': data_after[data_after.index >= t2]
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(title, fontsize=18, y=0.95)
    
    for i, (stage_name, df_stage) in enumerate(stages_data.items()):
        if not df_stage.empty:
            ctr_stage = df_stage[df_stage['chosen_arm'].isin(top_arms)].groupby('chosen_arm')['reward'].mean().reindex(top_arms)
            sns.barplot(x=ctr_stage.index, y=ctr_stage.values, ax=axes[i], order=top_arms, ci=None)
            axes[i].set_title(f"Stage: {stage_name}")
            axes[i].set_ylabel("Average Reward / CTR")
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.xlabel("Arm ID"); plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(save_path); plt.close()
    print(f"    ...saved to {save_path}")

def plot_context_pca(data, title, save_path, n_samples=10000):
    print(f"  Plotting: PCA of Context Vectors...")
    if not hasattr(data, 'context_vector'):
         print("    ...skipped, no 'context_vector' column found.")
         return
    sample = data.sample(n=min(len(data), n_samples), random_state=42)
    context_vectors = np.vstack(sample['context_vector'].values)
    pca = PCA(n_components=2).fit_transform(context_vectors)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca[:, 0], y=pca[:, 1], s=10, alpha=0.5, hue=sample['reward'])
    plt.title(title, fontsize=16)
    plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
    plt.grid(True); plt.tight_layout(); plt.legend(title='Reward')
    plt.savefig(save_path); plt.close()
    print(f"    ...saved to {save_path}")

def plot_movie_genre_distribution(handler, title, save_path):
    print("  Plotting: Movie Genre Distribution...")
    if not (hasattr(handler, 'all_genres') and hasattr(handler.data, 'context_vector')):
        print("    ...skipped, handler does not have required genre attributes.")
        return
    # This logic assumes the last 18 features of the context vector are genres
    genre_feature_dim = len(handler.all_genres)
    genre_vectors = np.vstack(handler.data['context_vector'].apply(lambda x: x[-genre_feature_dim:]).values)
    genre_counts = genre_vectors.sum(axis=0)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(handler.all_genres), y=genre_counts)
    plt.title(title, fontsize=16); plt.xlabel("Genre"); plt.ylabel("Total Count")
    plt.xticks(rotation=60, ha='right'); plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"    ...saved to {save_path}")

# --- Main Analysis Function ---
def analyze_scenario(handler, scenario_name, output_dir, is_drift_scenario=False):
    print("\n" + "="*50)
    print(f"Analyzing Scenario: {scenario_name}")
    print("="*50)
    
    data_before_drift = handler.data.copy()
    
    if is_drift_scenario:
        data_after_drift = create_drift_scenario(handler, scenario_name.lower())
    else:
        data_after_drift = data_before_drift

    t1, t2 = len(data_before_drift) // 3, 2 * len(data_before_drift) // 3

    # --- Generate Visualizations ---
    plot_reward_mean_with_drift(
        data_after_drift, 
        window_size=10000, 
        title=f"{scenario_name}: Rolling Reward Mean", 
        save_path=output_dir / f"fig_{scenario_name}_1_reward_mean.png",
        abrupt_drift_points=get_drift_points(scenario_name, len(data_after_drift)),
        gradual_drift_window=get_drift_window(scenario_name, len(data_after_drift))
    )
    
    plot_top_arm_ctr_stages(
        data_before_drift, data_after_drift, 10,
        title=f"{scenario_name}: Top-10 Arm CTR Across Stages",
        save_path=output_dir / f"fig_{scenario_name}_2_top_arm_ctr_stages.png",
        t1=t1, t2=t2
    )

    plot_context_pca(
        data_after_drift, f"{scenario_name}: Context PCA", 
        output_dir / f"fig_{scenario_name}_3_context_pca.png"
    )
    
    if isinstance(handler, MovieLensDataHandler) and 'ML' in scenario_name:
         # Only plot genre distribution for MovieLens scenarios
        plot_movie_genre_distribution(handler, f"{scenario_name}: Genre Distribution", output_dir / f"fig_{scenario_name}_4_genre_dist.png")

def main():
    output_dir = Path(__file__).parent.parent 
    data_dir = Path(__file__).parent.parent.parent
    
    # --- Load Data Handlers ---
    ml_handler = MovieLensDataHandler(data_path=data_dir / "ml-1m")
    obd_handler_bts = OBDDataHandler(data_path=data_dir / "obd", campaign="all", behavior_policy="bts")
    obd_handler_random = OBDDataHandler(data_path=data_dir / "obd", campaign="all", behavior_policy="random")

    # --- Run All Analyses ---
    # Baseline analyses
    analyze_scenario(ml_handler, "ML_Original", output_dir)
    analyze_scenario(obd_handler_bts, "OBD_BTS_Original", output_dir)
    analyze_scenario(obd_handler_random, "OBD_Random_Original", output_dir)
    
    # --- Drift Scenario Analyses ---
    analyze_scenario(ml_handler, "ML_Abrupt", output_dir, is_drift_scenario=True)
    analyze_scenario(ml_handler, "ML_Gradual", output_dir, is_drift_scenario=True)
    analyze_scenario(obd_handler_bts, "OBD_BTS_Abrupt", output_dir, is_drift_scenario=True)
    analyze_scenario(obd_handler_bts, "OBD_BTS_Gradual", output_dir, is_drift_scenario=True)

if __name__ == "__main__":
    main() 