import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# This script is in ucb_simulation_platform/notebooks/, so we need to adjust the path
# to find the ml-1m dataset, which we assume is in the parent directory of ucb_simulation_platform.
# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
DATA_DIR = 'ml-1m'
FIGURES_DIR = '../figures' # Relative to this script's location
MAX_K = 20 # Maximum number of clusters to test

def load_and_preprocess_users(data_dir: str) -> pd.DataFrame:
    """Loads and preprocesses the user data for clustering."""
    u_cols = ['user_id', 'gender', 'age', 'occupation', 'zipcode']
    users = pd.read_csv(
        os.path.join(data_dir, 'users.dat'), 
        sep='::', 
        names=u_cols, 
        engine='python', 
        encoding='latin-1'
    )
    
    # Feature Engineering: One-hot encode categorical features
    # We use 'age' and 'occupation' as they are strong demographic indicators.
    # Gender is binary, so it can be converted directly.
    users['gender_numeric'] = users['gender'].apply(lambda x: 1 if x == 'M' else 0)
    
    features_to_encode = ['age', 'occupation']
    user_features = pd.get_dummies(users, columns=features_to_encode, dtype=float)
    
    # Select final features for clustering
    # We drop original columns and zipcode (too noisy)
    features = user_features.drop(columns=['user_id', 'gender', 'zipcode'])
    
    # Scaling is crucial for K-Means
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features

def find_optimal_k(data: np.ndarray, max_k: int):
    """
    Uses the Elbow Method to find the optimal number of clusters (K).
    
    Args:
        data (np.ndarray): The preprocessed and scaled feature matrix.
        max_k (int): The maximum K to test.
        
    Returns:
        A dictionary mapping K to its Sum of Squared Errors (SSE).
    """
    sse = {}
    print(f"Calculating SSE for K from 2 to {max_k}...")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        sse[k] = kmeans.inertia_  # inertia_ is the SSE for the chosen k
        print(f"  K={k}, SSE={sse[k]:.2f}")
    return sse

def plot_elbow_method(sse: dict, save_path: str):
    """Plots the Elbow Method graph and saves it to a file."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    k_values = list(sse.keys())
    sse_values = list(sse.values())
    
    ax.plot(k_values, sse_values, 'bo-')
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=16)
    
    # Highlight the potential "elbow"
    # This is a heuristic, but often the point of maximum curvature.
    # We can programmatically find it, but visual inspection is key for the paper.
    # Let's highlight K=9 as per the prior research.
    if 9 in k_values:
        ax.axvline(x=9, color='red', linestyle='--', label='K=9 (from prior work)')
        ax.text(9.2, sse_values[k_values.index(9)], 'Potential Elbow', color='red', fontsize=12)

    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nElbow method plot saved to: {save_path}")

def main():
    """Main function to run the analysis."""
    print("--- Starting Optimal K Finder for MovieLens Users ---")
    
    # Check if data directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please ensure the 'ml-1m' dataset is in the correct location relative to the script.")
        return
        
    user_data_scaled = load_and_preprocess_users(DATA_DIR)
    sse_results = find_optimal_k(user_data_scaled, MAX_K)
    
    # Construct the save path relative to the script's location
    save_path = os.path.abspath(os.path.join(script_dir, FIGURES_DIR, 'elbow_method_plot.png'))
    
    plot_elbow_method(sse_results, save_path)
    
    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main() 