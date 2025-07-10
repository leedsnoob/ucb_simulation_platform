import logging
import numpy as np
import random
import os
import zipfile
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class BaseSimHandler(ABC):
    """
    Abstract base class for data simulation handlers.
    It defines the interface for loading data, building reward pools, 
    and providing environment properties.
    """
    def __init__(self, n_arms: int, seed: int):
        """
        Initializes the base handler.

        Args:
            n_arms (int): The number of arms in the environment.
            seed (int): The random seed for reproducibility.
        """
        self.n_arms = n_arms
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.reward_pools = {i: [] for i in range(n_arms)}
        self.true_mus = np.zeros(n_arms)
        self.true_stds = np.zeros(n_arms)
        self.best_arm_idx = 0
        
        self._build_reward_pools()
        self._calculate_true_statistics()

    @abstractmethod
    def _build_reward_pools(self):
        """
        An abstract method that subclasses must implement to process raw data
        and populate the internal `_reward_pools` dictionary.
        
        The `_reward_pools` should map each arm index to a list of its historical rewards.
        This method is also responsible for setting `self.n_arms` and `self.true_means`.
        """
        pass

    def _calculate_true_statistics(self):
        """
        Calculates the true mean and std dev for each arm from reward pools.
        """
        if not self.reward_pools:
             logging.warning("Reward pools are empty. Cannot calculate true statistics.")
             return
        
        for i in range(self.n_arms):
            if self.reward_pools.get(i):
                 self.true_mus[i] = np.mean(self.reward_pools[i])
                 self.true_stds[i] = np.std(self.reward_pools[i])
            else:
                 self.true_mus[i] = 0
                 self.true_stds[i] = 0

        self.best_arm_idx = np.argmax(self.true_mus)
        logging.info(f"True statistics calculated. Best arm is #{self.best_arm_idx} with µ* = {self.get_best_mu():.4f}")

    @abstractmethod
    def swap_extremes(self):
        """
        Abstract method for swapping the best and worst arms.
        Subclasses must implement this to modify their reward pools and true means.
        """
        pass

    def get_best_mu(self) -> float:
        """Returns the true mean of the best arm."""
        return self.true_mus[self.best_arm_idx]

    def get_n_arms(self) -> int:
        """Returns the number of arms."""
        return self.n_arms

    def get_true_means(self) -> np.ndarray:
        """Returns the true mean reward for each arm."""
        return self.true_mus

    def sample_reward(self, arm_index: int) -> float:
        """
        Samples a single reward from the historical pool for a given arm.

        Args:
            arm_index (int): The index of the arm to sample from.

        Returns:
            float: A randomly drawn reward value.
        
        Raises:
            KeyError: If the arm_index is invalid.
            ValueError: If the reward pool for the arm is empty.
        """
        if arm_index not in self._reward_pools:
            raise KeyError(f"Invalid arm index: {arm_index}. Arm does not exist.")
        
        pool = self._reward_pools[arm_index]
        if not pool:
            raise ValueError(f"Reward pool for arm {arm_index} is empty. Cannot sample.")
            
        return np.random.choice(pool)

    def get_reward_distribution(self) -> Dict[int, List[float]]:
        """Returns the complete dictionary of reward pools."""
        return self._reward_pools 