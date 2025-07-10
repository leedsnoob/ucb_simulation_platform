import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

from obp.dataset import OpenBanditDataset

class OBDDataHandler:
    """
    Handles loading and preprocessing of the Open Bandit Dataset (OBD) to simulate a bandit environment.
    """
    def __init__(self, data_path: str = 'obd', campaign: str = 'all', behavior_policy: str = 'random'):
        """
        Initializes the data handler by loading the OBD data.

        Args:
            data_path (str): The path to the directory containing the OBD campaign directories.
            campaign (str): The campaign to load ('all', 'men', or 'women').
            behavior_policy (str): The behavior policy that generated the data ('bts' or 'random').
        """
        self.data_path = Path(data_path)
        self.campaign = campaign
        self.behavior_policy = behavior_policy
        
        self._load_and_preprocess_data()
        self.current_index = 0
        self.context_dimension = self.data.iloc[0]['context_vector'].shape[0]
        print(f"Context vector dimension: {self.context_dimension}")

    def _load_and_preprocess_data(self):
        """
        Loads bandit feedback from the specified OBD campaign using the OBP library.
        """
        dataset = OpenBanditDataset(
            data_path=self.data_path, 
            campaign=self.campaign,
            behavior_policy=self.behavior_policy
        )
        
        # obp returns a dictionary of numpy arrays
        bandit_feedback = dataset.obtain_batch_bandit_feedback()

        # For consistency and ease of use, we convert it to a pandas DataFrame
        # Note: 'context' is already a numeric vector
        self.data = pd.DataFrame({
            'position': bandit_feedback['position'], # Keep position for sorting
            'context_vector': list(bandit_feedback['context']),
            'chosen_arm': bandit_feedback['action'],
            'reward': bandit_feedback['reward'],
        })
        
        # IMPORTANT: Sort by position to ensure deterministic order
        self.data.sort_values(by='position', inplace=True)
        self.data.reset_index(drop=True, inplace=True) # Reset index after sorting
        
        print(f"Loaded and deterministically sorted {len(self.data)} interactions from Open Bandit Dataset (campaign: {self.campaign}).")

    def __iter__(self):
        """
        Allows the handler to be used as an iterator.
        """
        self.current_index = 0
        return self

    def __next__(self):
        """
        Returns the next interaction in the stream.
        """
        if self.current_index < len(self.data):
            interaction = self.data.iloc[self.current_index]
            self.current_index += 1
            
            # The context vector from OBP is used directly
            context = {
                'vector': interaction['context_vector']
            }
            arm = interaction['chosen_arm']
            reward = interaction['reward']
            
            # In this simple replay, available_arms is just the one chosen arm
            return {'context': context, 'available_arms': [arm], 'chosen_arm': arm, 'reward': reward}
        else:
            raise StopIteration

    def reset(self):
        """
        Resets the iterator to the beginning of the data stream.
        """
        self.current_index = 0

    def __len__(self):
        """
        Returns the total number of interactions in the dataset.
        """
        return len(self.data)
