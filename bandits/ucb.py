import numpy as np
import random
from math import log, sqrt
from .base_bandit import BaseBandit


class UCBalgorithm:
    def __init__(self, num_arms, burn_in = 1, min_range = -float("inf"), max_range = float("inf"), epsilon = 0, delta = .1):
        self.num_arms = num_arms
        self.mean_estimators = [0 for _ in range(num_arms)]
        self.counts = [0 for _ in range(num_arms)]
        self.reward_sums = [0 for _ in range(num_arms)]
        self.burn_in = burn_in
        self.min_range = min_range
        self.max_range = max_range
        self.epsilon = epsilon
        self.delta = delta
        self.global_time_step = 0

    def reset(self):
        """Resets the algorithm's statistics to their initial state."""
        self.mean_estimators = [0 for _ in range(self.num_arms)]
        self.counts = [0 for _ in range(self.num_arms)]
        self.reward_sums = [0 for _ in range(self.num_arms)]
        self.global_time_step = 0

    def update_arm_statistics(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.reward_sums[arm_index] += reward
        self.mean_estimators[arm_index] = self.reward_sums[arm_index]/self.counts[arm_index] 
        self.global_time_step += 1

    def get_ucb_arm(self, confidence_radius, arm_info = None ):
        if sum(self.counts) <=  self.burn_in:
            ucb_arm_index = random.choice(range(self.num_arms))
        else:
            ucb_bonuses = [confidence_radius*np.sqrt(np.log((self.global_time_step+1.0)/self.delta)/(count + .0000000001)) for count in self.counts ]
            ucb_arm_values = [min(self.mean_estimators[i] + ucb_bonuses[i], self.max_range) for i in range(self.num_arms)]
            ucb_arm_values = np.array(ucb_arm_values)
            
            if np.random.random() <= self.epsilon:
                ucb_arm_index = np.random.choice(range(self.num_arms))
            else:
                ucb_arm_index = np.random.choice(np.flatnonzero(ucb_arm_values == ucb_arm_values.max()))

        return ucb_arm_index

    def get_arm(self, parameter, arm_info = None):
        return self.get_ucb_arm(parameter, arm_info = arm_info)


class UCB1(BaseBandit):
    """
    Upper Confidence Bound 1 (UCB1) algorithm.
    """
    def __init__(self, n_arms: int, alpha: float = 1.0, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.alpha = alpha
        self.arm_counts = np.zeros(n_arms, dtype=int)
        self.estimated_rewards = np.zeros(n_arms, dtype=float)

    def select_arm(self, context=None) -> int:
        # Initial exploration: play each arm once
        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0:
                return arm
        
        # UCB calculation
        ucb_values = np.zeros(self.n_arms)
        
        # self.t starts from 0, so we use t+1 for log to avoid log(0)
        # It represents the total number of pulls so far.
        total_pulls = self.t if self.t > 0 else 1

        for arm in range(self.n_arms):
            bonus = self.alpha * sqrt(log(total_pulls) / self.arm_counts[arm])
            ucb_values[arm] = self.estimated_rewards[arm] + bonus
            
        return np.argmax(ucb_values)

    def update(self, arm_idx: int, reward: float) -> None:
        n = self.arm_counts[arm_idx]
        
        # Update estimated reward using an incremental mean
        old_value = self.estimated_rewards[arm_idx]
        self.estimated_rewards[arm_idx] = (n * old_value + reward) / (n + 1)
        
        self.arm_counts[arm_idx] += 1
        self.t += 1

    def reset(self):
        super().reset()
        self.arm_counts.fill(0)
        self.estimated_rewards.fill(0)
