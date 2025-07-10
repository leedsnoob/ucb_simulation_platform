import numpy as np
from math import log, sqrt
from collections import deque

from .base_bandit import BaseBandit

class UCB_SW(BaseBandit):
    """
    Sliding Window Upper Confidence Bound (SW-UCB) algorithm.
    The window size can be dynamically calculated based on the horizon.
    """
    def __init__(self, n_arms: int, alpha: float = 1.0, window_size: int = None, 
                 horizon: int = None, drift_config: list = None, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.alpha = alpha
        
        if window_size:
            self.window_size = window_size
        elif horizon:
            # Dynamically calculate window size based on SMPyBandits' heuristic
            # Formula: C * sqrt(T * log(T) / L)
            # We use C=2 as a reasonable default constant.
            num_drifts = len(drift_config) if drift_config else 1
            self.window_size = int(2 * np.sqrt(horizon * np.log(horizon) / num_drifts))
            print(f"UCB_SW dynamically calculated window size: {self.window_size}")
        else:
            raise ValueError("Either 'window_size' or 'horizon' must be provided.")

        # This tracks total pulls for the initial round-robin
        self.arm_counts = np.zeros(n_arms, dtype=int)
        
        # This deque stores (arm, reward) tuples for the global window
        self.history_window = deque(maxlen=self.window_size)
        
        # These track the counts and rewards *within the window only*
        self.window_arm_counts = np.zeros(n_arms, dtype=int)
        self.window_arm_rewards = np.zeros(n_arms, dtype=float)

    def select_arm(self, context=None) -> int:
        """Selects an arm based on UCB calculated from the global sliding window."""
        # Use the overall arm_counts for the initial exploration phase
        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0:
                return arm
        
        total_counts_in_window = len(self.history_window)
        if total_counts_in_window == 0:
            return np.random.randint(self.n_arms)

        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            n_k = self.window_arm_counts[arm]
            if n_k == 0:
                ucb_values[arm] = float('inf') 
                continue
            
            mean_reward = self.window_arm_rewards[arm] / n_k
            bonus = self.alpha * np.sqrt(np.log(total_counts_in_window) / n_k)
            ucb_values[arm] = mean_reward + bonus
            
        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float) -> None:
        """Updates the global sliding window and the total arm counts."""
        self.t += 1
        self.arm_counts[arm] += 1

        # Update window state
        if len(self.history_window) == self.window_size:
            oldest_arm, oldest_reward = self.history_window[0]
            self.window_arm_counts[oldest_arm] -= 1
            self.window_arm_rewards[oldest_arm] -= oldest_reward

        self.history_window.append((arm, reward))
        self.window_arm_counts[arm] += 1
        self.window_arm_rewards[arm] += reward
        
    def reset(self) -> None:
        """Resets the agent to its initial state."""
        super().reset()
        self.arm_counts.fill(0)
        self.history_window.clear()
        self.window_arm_counts.fill(0)
        self.window_arm_rewards.fill(0) 