import numpy as np
from math import log, sqrt
from .base_bandit import BaseBandit

class UCB_D(BaseBandit):
    """
    The Discounted Upper Confidence Bound (UCB-D) algorithm.
    It discounts past rewards to adapt to non-stationary environments by decaying
    the effective counts of successes and failures for each arm, inspired by
    the logic in "Taming Non-stationary Bandits: A Bayesian Approach".
    """
    def __init__(self, n_arms: int, alpha: float = 1.0, gamma: float = 0.99, **kwargs):
        """
        Initializes the Discounted UCB algorithm.

        Parameters
        ----------
        n_arms : int
            Number of arms.
        alpha : float
            Exploration parameter. A higher value leads to more exploration.
        gamma : float
            Discounting factor, between 0 and 1. A lower value means
            the algorithm "forgets" past rewards faster.
        """
        super().__init__(n_arms, **kwargs)
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.alpha = alpha
        self.gamma = gamma

        # CORRECTED: Use generic reward sum and counts to support non-binary rewards.
        self.discounted_reward_sum = np.zeros(n_arms)
        self.discounted_counts = np.zeros(n_arms)

    def select_arm(self, context=None) -> int:
        """Selects an arm according to the Discounted UCB policy."""
        # Use a small epsilon to avoid division by zero
        safe_counts = self.discounted_counts + 1e-8
        
        # Initial exploration: play each arm once to initialize
        for arm in range(self.n_arms):
            if self.discounted_counts[arm] == 0:
                return arm
        
        # After initialization, calculate UCB values based on discounted stats
        total_counts_all_arms = np.sum(self.discounted_counts)
        if total_counts_all_arms == 0:
            return np.random.randint(self.n_arms)
            
        ucb_values = np.zeros(self.n_arms)
        
        log_numerator = total_counts_all_arms if total_counts_all_arms > 0 else 1
        
        for arm in range(self.n_arms):
            arm_total_counts = self.discounted_counts[arm]
            if arm_total_counts < 1e-8:
                ucb_values[arm] = float('inf')
                continue

            # Calculate estimated average reward from discounted stats
            average_reward = self.discounted_reward_sum[arm] / arm_total_counts
            
            # Calculate the confidence bound
            bonus = self.alpha * sqrt(log(log_numerator) / arm_total_counts)
            
            ucb_values[arm] = average_reward + bonus
            
        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float) -> None:
        """
        Updates the discounted statistics for all arms.
        """
        self.t += 1
        # 1. Discount all arms' statistics
        self.discounted_reward_sum *= self.gamma
        self.discounted_counts *= self.gamma

        # 2. Update the statistics for the chosen arm
        self.discounted_reward_sum[arm] += reward
        self.discounted_counts[arm] += 1

    def reset(self) -> None:
        """Resets the agent to its initial state."""
        super().reset()
        self.discounted_reward_sum.fill(0)
        self.discounted_counts.fill(0)

    @property
    def values(self) -> np.ndarray:
        """Returns the current estimated values of the arms."""
        safe_counts = self.discounted_counts + 1e-8
        return self.discounted_reward_sum / safe_counts 