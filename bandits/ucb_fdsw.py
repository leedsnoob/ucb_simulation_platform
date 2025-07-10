import numpy as np
from math import log, sqrt
from collections import deque

from .base_bandit import BaseBandit

class FDSW_UCB(BaseBandit):
    """
    f-Discounted and Sliding-Window UCB (FDSW-UCB) algorithm.
    
    This algorithm maintains two separate estimates for each arm:
    1.  A 'historic trace' using a discounting factor (gamma).
    2.  A 'hot trace' using a sliding window of the most recent rewards.
    An aggregation function 'f' combines the UCB scores from both traces.
    """
    def __init__(self, n_arms: int, alpha: float = 1.0, gamma: float = 0.99, 
                 window_size: int = None, horizon: int = None, drift_config: list = None,
                 agg_function: str = 'mean', **kwargs):
        super().__init__(n_arms, **kwargs)
        if agg_function not in ['min', 'max', 'mean']:
            raise ValueError("agg_function must be 'min', 'max', or 'mean'.")
        
        self.alpha = alpha
        self.gamma = gamma
        self.agg_function = agg_function

        if window_size:
            self.window_size = window_size
        elif horizon:
            # Dynamically calculate window size based on SMPyBandits' heuristic
            num_drifts = len(drift_config) if drift_config else 1
            self.window_size = int(2 * np.sqrt(horizon * np.log(horizon) / num_drifts))
            print(f"FDSW_UCB dynamically calculated window size: {self.window_size}")
        else:
            raise ValueError("Either 'window_size' or 'horizon' must be provided.")

        # --- State for Discounted Trace ---
        self.d_reward_sum = np.zeros(n_arms)
        self.d_counts = np.zeros(n_arms)

        # --- State for Sliding Window Trace (Global Window) ---
        self.sw_history = deque(maxlen=self.window_size) # Global history of (arm, reward)
        self.sw_counts = np.zeros(n_arms, dtype=int)
        self.sw_reward_sums = np.zeros(n_arms, dtype=float)

    def select_arm(self, context=None) -> int:
        # Initial exploration
        for arm in range(self.n_arms):
            # Check based on discounted counts as it represents the "ever-pulled" memory
            if (self.d_counts[arm]) == 0:
                return arm

        ucb_d = np.zeros(self.n_arms)
        ucb_sw = np.zeros(self.n_arms)
        
        # Calculate UCB for both views
        d_total_counts = self.d_counts
        d_total_counts_all_arms = np.sum(d_total_counts)
        sw_total_counts = len(self.sw_history)

        for arm in range(self.n_arms):
            # 1. Discounted UCB
            d_arm_counts = self.d_counts[arm]
            if d_arm_counts < 1e-8: # Epsilon check for safety
                ucb_d[arm] = float('inf')
            else:
                mean_d = self.d_reward_sum[arm] / d_arm_counts
                # CORRECTED: The log numerator should be the sum of discounted counts.
                log_numerator_d = d_total_counts_all_arms if d_total_counts_all_arms > 0 else 1
                bonus_d = self.alpha * np.sqrt(np.log(log_numerator_d) / d_arm_counts)
                ucb_d[arm] = mean_d + bonus_d

            # 2. Sliding Window UCB
            sw_arm_counts = self.sw_counts[arm]
            if sw_arm_counts == 0:
                 ucb_sw[arm] = float('inf')
            else:
                mean_sw = self.sw_reward_sums[arm] / sw_arm_counts
                # Note: log uses total pulls from the whole history, but count is from window
                # This is a common practice in combined algorithms.
                bonus_sw = self.alpha * np.sqrt(np.log(sw_total_counts if sw_total_counts > 0 else 1) / sw_arm_counts)
                ucb_sw[arm] = mean_sw + bonus_sw
        
        # --- Aggregation ---
        if self.agg_function == 'min':
            final_ucb = np.minimum(ucb_d, ucb_sw)
        elif self.agg_function == 'max':
            final_ucb = np.maximum(ucb_d, ucb_sw)
        else: # 'mean'
            final_ucb = (ucb_d + ucb_sw) / 2.0
            
        return np.argmax(final_ucb)

    def update(self, arm: int, reward: float) -> None:
        self.t += 1 # Increment t first
        
        # --- 1. Update Discounted History ---
        self.d_reward_sum *= self.gamma
        self.d_counts *= self.gamma
        self.d_reward_sum[arm] += reward
        self.d_counts[arm] += 1
        
        # --- 2. Update Sliding Window History ---
        # Check if the window is full to remove the oldest element's stats
        if len(self.sw_history) == self.window_size:
            oldest_arm, oldest_reward = self.sw_history[0]
            self.sw_counts[oldest_arm] -= 1
            self.sw_reward_sums[oldest_arm] -= oldest_reward

        # Add the new observation
        self.sw_history.append((arm, reward))
        self.sw_counts[arm] += 1
        self.sw_reward_sums[arm] += reward
            
    def reset(self) -> None:
        """Resets the agent to its initial state."""
        super().reset()
        self.d_reward_sum.fill(0)
        self.d_counts.fill(0)
        self.sw_history.clear()
        self.sw_counts.fill(0)
        self.sw_reward_sums.fill(0) 