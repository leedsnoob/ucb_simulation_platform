import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class DriftInjector:
    """
    A class to inject concept drift into a bandit dataset provided as a pandas DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the injector with the original bandit data.

        Args:
            data (pd.DataFrame): The original bandit feedback data. 
                                 Must contain 'chosen_arm' and 'reward' columns.
        """
        self.data = data.copy()

    def inject_abrupt_drift(self, position: int, arms_to_drift: list, drift_type: str = 'invert'):
        """
        Injects an abrupt drift at a specific position in the data stream.

        Args:
            position (int): The interaction index after which the drift occurs.
            arms_to_drift (list): A list of arm IDs that should be affected by the drift.
            drift_type (str): The type of drift. Currently supports 'invert' which flips the reward.

        Returns:
            DriftInjector: The injector instance with the injected drift.
        """
        if not (0 < position < len(self.data)):
            raise ValueError("Drift position must be within the data's time range.")

        print(f"Injecting abrupt drift of type '{drift_type}' at position {position} for {len(arms_to_drift)} arms.")

        # Identify the part of the data that will be affected by the drift
        drift_segment_filter = (self.data.index >= position) & (self.data['chosen_arm'].isin(arms_to_drift))
        
        if drift_type == 'invert':
            # Invert the reward (1 becomes 0, 0 becomes 1)
            self.data.loc[drift_segment_filter, 'reward'] = 1 - self.data.loc[drift_segment_filter, 'reward']
        else:
            raise NotImplementedError(f"Drift type '{drift_type}' is not implemented.")
            
        print("Abrupt drift injection complete.")
        return self

    def inject_gradual_drift(self, start_position: int, end_position: int, arms_to_drift: list, drift_type: str = 'invert'):
        """
        Injects a gradual drift over a specified window by progressively changing the reward probability.

        Args:
            start_position (int): The interaction index where the drift begins.
            end_position (int): The interaction index where the drift completes.
            arms_to_drift (list): A list of arm IDs that should be affected by the drift.
            drift_type (str): The type of drift. Currently supports 'invert'.

        Returns:
            DriftInjector: The injector instance with the injected gradual drift.
        """
        if not (0 <= start_position < end_position < len(self.data)):
            raise ValueError("Invalid drift window specified.")

        print(f"Injecting gradual drift from position {start_position} to {end_position}.")

        # Filter the segment of data where drift will occur
        drift_window_filter = (self.data.index >= start_position) & (self.data.index < end_position)
        drift_arms_filter = self.data['chosen_arm'].isin(arms_to_drift)
        target_indices = self.data[drift_window_filter & drift_arms_filter].index

        window_size = end_position - start_position

        # Use tqdm for a progress bar as this is an iterative process
        for idx in tqdm(target_indices, desc="Injecting Gradual Drift"):
            # Calculate the progress within the drift window (0.0 to 1.0)
            progress = (idx - start_position) / window_size
            
            # The probability of flipping the reward increases with progress
            if np.random.rand() < progress:
                if drift_type == 'invert':
                    self.data.loc[idx, 'reward'] = 1 - self.data.loc[idx, 'reward']
                else:
                    raise NotImplementedError(f"Drift type '{drift_type}' is not implemented.")
        
        print("Gradual drift injection complete.")
        return self

    def get_drifted_data(self):
        """
        Returns the final modified DataFrame.
        """
        return self.data 