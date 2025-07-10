import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import random
from tqdm import tqdm
import logging
import zipfile

from ucb_simulation_platform.data.base_sim_handler import BaseSimHandler


class MovieLensSimHandler(BaseSimHandler):
    """
    A simulation handler for the MovieLens-1M dataset.
    
    This class performs the following key steps upon initialization:
    1.  Loads user data (`users.dat`).
    2.  Performs K-Means clustering on user features to define a set of 'arms' (user profiles).
    3.  Loads ratings data (`ratings.dat`).
    4.  Builds a 'reward pool' for each arm, containing all real ratings from users in that cluster.
    5.  Pre-computes the true mean reward (`mu`) for each arm, enabling the calculation of true regret.
    """
    
    def __init__(self, seed: int, k: int = 9):
        # Use absolute path for robustness
        self.data_dir = '/root/autodl-tmp/MAB/ml-1m'
        self.seed = seed
        self.k = k
        self.rng = np.random.default_rng(self.seed)
        
        print(f"--- Initializing MovieLensSimHandler with K={k} ---")
        
        if not os.path.isdir(self.data_dir):
            self._download_and_unzip_movielens()
            
        self.users_df, self.user_features_scaled = self._load_and_preprocess_users()
        self.ratings_df = self._load_ratings()
        self.item_df = self._load_items()

        # Cluster users to define arms
        self.users_df['arm'] = self._cluster_users()
        self.n_arms = self.users_df['arm'].nunique()

        # Now that all data is ready, call the parent constructor
        super().__init__(n_arms=self.n_arms, seed=seed)
        
        # --- Gradual Drift State ---
        self.is_drifting = False
        self.drift_start_t = 0
        self.drift_duration = 0
        self.drift_arm_A = -1
        self.drift_arm_B = -1
        self.drift_pool_A = []
        self.drift_pool_B = []
        
        logging.info(f"MovieLens reward pools built for {self.n_arms} arms.")

    def _download_and_unzip_movielens(self):
        """Downloads and unzips the MovieLens 1M dataset."""
        # Use absolute path for the zip file as well
        zip_path = '/root/autodl-tmp/MAB/ml-1m.zip'
        extract_path = '/root/autodl-tmp/MAB/'
        
        if not os.path.exists(zip_path):
            # ... (rest of the download logic uses these corrected paths)
            pass # Placeholder for actual download logic
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # The unzipped folder is 'ml-1m', which becomes our data_dir
        self.data_dir = data_dir 

    def _load_and_preprocess_users(self) -> pd.DataFrame:
        """Loads and preprocesses the users.dat file."""
        u_cols = ['user_id', 'gender', 'age', 'occupation', 'zipcode']
        users = pd.read_csv(
            os.path.join(self.data_dir, 'users.dat'), 
            sep='::', names=u_cols, engine='python', encoding='latin-1'
        )
        
        # Feature Engineering
        users['gender_numeric'] = users['gender'].apply(lambda x: 1 if x == 'M' else 0)
        features_to_encode = ['age', 'occupation']
        user_features = pd.get_dummies(users, columns=features_to_encode, dtype=float)
        features = user_features.drop(columns=['user_id', 'gender', 'zipcode'])
        
        # Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
        users['cluster_id'] = kmeans.fit_predict(scaled_features)
        
        return users[['user_id', 'cluster_id']], scaled_features

    def _load_ratings(self) -> pd.DataFrame:
        """Loads the ratings.dat file."""
        r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(
            os.path.join(self.data_dir, 'ratings.dat'), 
            sep='::', names=r_cols, engine='python', encoding='latin-1'
        )
        return ratings

    def _load_items(self) -> pd.DataFrame:
        """Loads the movies.dat file."""
        i_cols = ['movie_id', 'title', 'genre']
        items = pd.read_csv(
            os.path.join(self.data_dir, 'movies.dat'), 
            sep='::', names=i_cols, engine='python', encoding='latin-1'
        )
        return items

    def _cluster_users(self) -> pd.Series:
        """Clusters users based on pre-processed features."""
        # Use the scaled features from _load_and_preprocess_users
        kmeans = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
        return kmeans.fit_predict(self.user_features_scaled)

    def _build_reward_pools(self):
        """Builds a dictionary of reward pools for each user cluster."""
        logging.info("Building reward pools from user clusters...")
        
        merged_df = pd.merge(self.ratings_df, self.users_df, on='user_id')
        
        # Group by cluster and aggregate ratings into lists
        reward_pools_df = merged_df.groupby('arm')['rating'].apply(list).reset_index(name='rewards')
        
        # Populate the reward_pools dictionary
        for _, row in reward_pools_df.iterrows():
            self.reward_pools[row['arm']] = row['rewards']
    
    # The _calculate_true_statistics method is now handled by the base class,
    # so we don't need to define get_true_means, get_best_mu etc. here anymore.

    def sample_reward(self, arm_index: int, t: int) -> float:
        """
        Samples a single reward. If a gradual drift is active, it may interpolate
        rewards between the drifting arms.
        """
        if self.is_drifting and self.drift_start_t <= t < self.drift_start_t + self.drift_duration:
            # --- We are in a gradual drift phase ---
            progress = (t - self.drift_start_t) / self.drift_duration
            
            if arm_index == self.drift_arm_A:
                # Sample from B's pool with probability 'progress'
                if self.rng.random() < progress:
                    return self.rng.choice(self.drift_pool_B)
                else:
                    return self.rng.choice(self.drift_pool_A)

            elif arm_index == self.drift_arm_B:
                # Sample from A's pool with probability 'progress'
                if self.rng.random() < progress:
                    return self.rng.choice(self.drift_pool_A)
                else:
                    return self.rng.choice(self.drift_pool_B)

        # --- Default behavior: no drift or arm not involved in drift ---
        if arm_index not in self.reward_pools or not self.reward_pools[arm_index]:
            return 0.0 # Return 0 if arm is invalid or has no rewards
        return self.rng.choice(self.reward_pools[arm_index])


    def init_gradual_drift(self, start: int, duration: int):
        """
        Initializes a gradual drift by swapping the best and worst arms over a period.
        """
        if self.n_arms < 2:
            return

        logging.info(f"Initializing gradual drift from t={start} for {duration} steps.")
        self.is_drifting = True
        self.drift_start_t = start
        self.drift_duration = duration
        
        # Identify best and worst arms at the moment of initialization
        sorted_indices = np.argsort(self.true_mus)
        self.drift_arm_A = sorted_indices[-1] # The best arm
        self.drift_arm_B = sorted_indices[0]  # The worst arm
        
        # Store a snapshot of their reward pools at the beginning of the drift
        self.drift_pool_A = list(self.reward_pools[self.drift_arm_A])
        self.drift_pool_B = list(self.reward_pools[self.drift_arm_B])

    def get_true_means(self) -> dict:
        """Returns the dictionary of pre-computed true mean rewards."""
        return self.true_mus

    def get_mu_star(self) -> float:
        """Returns the pre-computed optimal mean reward."""
        return self.mu_star 

    def get_user_clusters(self) -> pd.DataFrame:
        """Returns the dataframe of users with their assigned cluster."""
        return self.users_df

    def get_ratings(self) -> pd.DataFrame:
        """Returns the dataframe of ratings."""
        return self.ratings_df 

    def swap_extremes(self):
        """
        Swaps the reward pools and true means of the top-2 and bottom-2 arms.
        This method directly modifies the internal state of the handler.
        """
        if self.n_arms < 4:
            return # Not enough arms to swap

        # Use the extreme arm indices calculated by the environment
        # Note: This assumes the environment has already called _update_extreme_arms
        # This is a bit of a dependency smell, but we'll stick to it for now.
        # A better design might pass the indices to be swapped as arguments.
        
        # We need to get the arm indices from the environment, which is tricky.
        # Let's re-calculate them here based on current mus for robustness.
        sorted_indices = np.argsort(self.true_mus)
        bottom_arms = sorted_indices[:2]
        top_arms = sorted_indices[-2:]

        # Swap reward pools
        (self.reward_pools[top_arms[0]], self.reward_pools[bottom_arms[0]]) = \
        (self.reward_pools[bottom_arms[0]], self.reward_pools[top_arms[0]])
        
        (self.reward_pools[top_arms[1]], self.reward_pools[bottom_arms[1]]) = \
        (self.reward_pools[bottom_arms[1]], self.reward_pools[top_arms[1]])

        # After swapping pools, we MUST recalculate the true statistics
        self._calculate_true_statistics() 