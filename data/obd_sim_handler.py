import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
import logging

from ucb_simulation_platform.data.base_sim_handler import BaseSimHandler


class OBDSimHandler(BaseSimHandler):
    """
    A simulation handler for the Open Bandit Dataset (OBD).
    
    This class defines 'arms' as the unique item IDs found in the dataset.
    It performs the following key steps upon initialization:
    1.  Loads the main interaction data (`all.csv` from the 'random' policy).
    2.  Identifies all unique `item_id`s to serve as the arms.
    3.  Builds a 'reward pool' for each item arm from the historical reward data (0s and 1s).
    4.  Pre-computes the true mean reward (`mu`) for each arm.
    """
    
    def __init__(self, seed: int, policy: str = 'random'):
        """
        Initializes the handler for the Open Bandit Dataset.

        Args:
            seed (int): The random seed.
            policy (str): The policy directory to use ('random' or 'bts').
        """
        self.data_dir = '/root/autodl-tmp/MAB/obd'
        self.seed = seed
        self.policy = policy
        
        if not os.path.isdir(os.path.join(self.data_dir, self.policy)):
            raise FileNotFoundError(f"Policy directory not found: '{os.path.join(self.data_dir, self.policy)}'")
            
        self.all_df = self._load_data()
        self.item_ids = sorted(self.all_df['item_id'].unique())
        self.arm_mapping = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.n_arms = len(self.item_ids)
        
        super().__init__(n_arms=self.n_arms, seed=seed)
        logging.info(f"OBD reward pools built for {self.n_arms} arms from '{self.policy}' policy.")

    def _load_data(self) -> pd.DataFrame:
        """Loads the `all.csv` from the specified policy directory."""
        print(f"--- Initializing OBDSimHandler from '{self.data_dir}/{self.policy}' ---")
        
        self.data_path = os.path.join(self.data_dir, self.policy, 'all', 'all.csv')

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: '{self.data_path}'")
        
        data_df = pd.read_csv(self.data_path)
        return data_df

    def _build_reward_pools(self):
        """Builds reward pools for each item_id from the OBD dataset."""
        logging.info("Building OBD reward pools...")
        
        # item_id is directly used as the arm
        grouped = self.all_df.groupby('item_id')['click']
        
        # Create a dictionary of reward pools for each arm index
        # The arm_mapping converts item_id to a 0-based index
        for item_id, rewards in grouped:
            arm_idx = self.arm_mapping[item_id]
            self.reward_pools[arm_idx] = rewards.tolist()
            
    # The _calculate_true_statistics method is now handled by the base class.
    
    def swap_extremes(self):
        """
        Swaps the reward pools and true means of the top-2 and bottom-2 arms.
        """
        if self.n_arms < 4:
            return

        sorted_indices = np.argsort(self.true_mus)
        bottom_arms = sorted_indices[:2]
        top_arms = sorted_indices[-2:]

        # Swap reward pools
        (self.reward_pools[top_arms[0]], self.reward_pools[bottom_arms[0]]) = \
        (self.reward_pools[bottom_arms[0]], self.reward_pools[top_arms[0]])
        
        (self.reward_pools[top_arms[1]], self.reward_pools[bottom_arms[1]]) = \
        (self.reward_pools[bottom_arms[1]], self.reward_pools[top_arms[1]])

        # After swapping pools, recalculate the true statistics
        self._calculate_true_statistics()
    
    def sample_reward(self, arm_index: int) -> int:
        """Samples a single reward from the specified arm's reward pool."""
        if arm_index not in self.reward_pools or not self.reward_pools[arm_index]:
            return 0.0
        return self.random_generator.choice(self.reward_pools[arm_index])

    def get_true_means(self) -> dict:
        """Returns the dictionary of pre-computed true mean rewards."""
        return self.true_mus

    def get_mu_star(self) -> float:
        """Returns the pre-computed optimal mean reward."""
        return self.mu_star 