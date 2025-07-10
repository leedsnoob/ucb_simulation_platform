import numpy as np
import copy
import pandas as pd
from ucb_simulation_platform.data.movielens_sim_handler import MovieLensSimHandler
from ucb_simulation_platform.data.obd_sim_handler import OBDSimHandler
from ucb_simulation_platform.data.base_sim_handler import BaseSimHandler
import logging


class BaseEnvironment:
    """环境基类，定义了与 Bandit 算法交互的核心接口。"""

    def __init__(self, handler: BaseSimHandler, seed: int):
        """
        初始化环境。

        Args:
            handler (BaseSimHandler): 数据和模拟逻辑的处理器。
            seed (int): 随机种子。
        """
        self.handler = handler
        self.n_arms = handler.n_arms
        self.reward_pools = handler.reward_pools
        self.true_mus = handler.true_mus
        self.true_stds = handler.true_stds
        # Remove best_mu and best_arm_idx from here, they will be dynamic
        self.best_mu = handler.get_best_mu() # Correctly call the existing method
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """重置环境状态。"""
        self.t = 0
        self.arm_counts = np.zeros(self.n_arms, dtype=int)

    def step(self, arm_idx: int, t: int) -> float:
        """
        执行一个时间步：拉动一个臂并获得奖励。

        Args:
            arm_idx (int): 被拉动的臂的索引。

        Returns:
            float: 获得的奖励。
        """
        if arm_idx < 0 or arm_idx >= self.n_arms:
            raise ValueError("Invalid arm index.")

        self.arm_counts[arm_idx] += 1
        self.t += 1
        
        # 奖励是直接从真实均值数组中获取的
        reward = self.pull_arm(arm_idx)
        
        return reward

    def pull_arm(self, arm_idx: int) -> float:
        """从指定臂的奖励池中随机抽取一个奖励。"""
        # Abrupt & Gradual drift is handled by directly changing true_mus
        # and reward_pools in the handler.
        
        # 从奖励池中随机抽样一个奖励
        reward_pool = self.reward_pools[arm_idx]
        if not reward_pool:
            # 在我们的设计中，奖励池不应为空。
            # 但作为备用，如果为空，则返回真实均值。
            return self.true_mus[arm_idx]
        return self.rng.choice(reward_pool)

    def get_true_means(self):
        """返回所有臂的真实平均奖励，确保返回的是副本以防外部修改。"""
        return self.true_mus.copy()

    def get_best_mu(self) -> float:
        """Returns the current best mu stored in the environment."""
        # This now returns the env's view of mu_star, which gets updated on drift
        return self.best_mu

class StationaryEnvironment(BaseEnvironment):
    """
    Represents a stationary (non-changing) bandit environment.
    
    This class serves as a base for all simulation environments. It gets its
    properties (number of arms, true rewards) from a data handler.
    """
    
    def __init__(self, handler: BaseSimHandler, seed: int):
        """
        Initializes the environment using a data handler.
        
        Args:
            data_handler: An instance of a simulation data handler 
                          (e.g., MovieLensSimHandler).
        """
        print(f"--- Initializing StationaryEnvironment ---")
        super().__init__(handler, seed)
        self.handler = handler
        self.n_arms = self.handler.n_arms
        self.true_mus = self.handler.get_true_means()
        self.mu_star = self.handler.get_mu_star()
        print("--- Environment Initialized ---")

    def step(self, arm_index: int, t: int) -> float:
        """Takes an action and returns a reward, advancing the environment state."""
        # Check for and apply abrupt drifts at the current timestep
        if t in self.drift_events:
            for event_func in self.drift_events[t]:
                event_func()
                # After drift, we must update the optimal arm's value
                self.mu_star = self.handler.get_mu_star()

        # The handler now needs the timestep for gradual drifts
        reward = self.handler.sample_reward(arm_index, t)
        return reward

    def get_true_means(self) -> dict:
        """Returns the true mean rewards of all arms."""
        return self.true_mus

    def get_mu_star(self) -> float:
        """Returns the optimal true mean reward."""
        return self.mu_star


class NonStationaryEnvironment(BaseEnvironment):
    """
    Represents a non-stationary environment where reward distributions can change.
    """
    def __init__(self, handler: BaseSimHandler, seed: int):
        super().__init__(handler, seed)
        # self.true_mus is already initialized in BaseEnvironment
        self.drift_events = []
        self._update_extreme_arms()

    def init_drift(self, drift_config: list):
        """Initializes the drift events from a configuration list."""
        self.drift_events = sorted(drift_config, key=lambda x: x.get('position', x.get('start', float('inf'))))
        
        # Pre-calculate gradual drift values if any
        for event in self.drift_events:
            if event['type'] == 'gradual_linear':
                if not hasattr(self.handler, 'init_gradual_drift'):
                    raise NotImplementedError("The provided handler does not support 'init_gradual_drift'.")
                self.handler.init_gradual_drift(start=event['start'], duration=event['duration'])

    def check_and_apply_drift(self):
        """Checks if a drift event should occur at the current timestep and applies it."""
        if not self.drift_events:
            return

        # Check for the next event based on current time `self.t`
        next_event_pos = self.drift_events[0].get('position', self.drift_events[0].get('start', float('inf')))
        
        if self.t >= next_event_pos:
            event = self.drift_events.pop(0)
            event_type = event.get('type')
            
            logging.info(f"Drift event '{event_type}' triggered at t={self.t}")

            if event_type == 'swap_extremes':
                self._swap_extremes()
            # Gradual drift is handled implicitly by the handler now
            # No need for explicit step-by-step update in environment
            
    def _recompute_mus_from_pools(self):
        """
        Private helper to re-calculate all true_mus based on the current state
        of the reward_pools. This is essential after a drift.
        """
        for arm_id, pool in self.reward_pools.items():
            if not pool:
                self.true_mus[arm_id] = 0.0
            else:
                self.true_mus[arm_id] = np.mean(pool)

    def _swap_extremes(self):
        """Swaps the reward pools and true means of the best/worst arms."""
        logging.info(f"Swapping extreme arms at t={self.t}")
        self.handler.swap_extremes()
        # Sync the environment's state with the handler's updated state
        self.true_mus = self.handler.get_true_means()
        self.best_mu = self.handler.get_best_mu()
        self._update_extreme_arms()

    def _update_extreme_arms(self):
        """
        Identifies the indices of the top-2 best and bottom-2 worst arms
        based on the current true mean rewards.
        """
        if self.n_arms < 4:
            # Not enough arms to perform a top-2/bottom-2 swap
            self.bottom_arms = []
            self.top_arms = []
            return
            
        # Use numpy's argsort to get the indices that would sort the array
        sorted_indices = np.argsort(self.true_mus)
        
        self.bottom_arms = sorted_indices[:2]
        self.top_arms = sorted_indices[-2:]
        logging.info(f"Extreme arms updated. Bottom-2: {self.bottom_arms}, Top-2: {self.top_arms}")
        
    def step(self, arm_idx: int, t: int) -> float:
        """
        Execute a step, applying drift before returning the reward.
        """
        self.check_and_apply_drift()
        return super().step(arm_idx, t)
        
    def get_true_means(self) -> dict:
        """
        Overrides the parent method to ensure a *copy* of the 
        mutated internal state is returned.
        """
        return self.true_mus.copy() 