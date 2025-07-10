import numpy as np
from abc import ABC, abstractmethod

class BaseBandit(ABC):
    """
    Abstract base class for all bandit algorithms.
    """
    
    def __init__(self, n_arms: int, **kwargs):
        self.n_arms = n_arms
        self.t = 0
        # Each subclass is responsible for its own state tracking.
        # This base class only provides the interface.

    @abstractmethod
    def select_arm(self):
        """Select an arm to pull."""
        raise NotImplementedError

    @abstractmethod
    def update(self, arm_idx: int, reward: float):
        """Update the algorithm's state with the observed reward."""
        raise NotImplementedError

    def reset(self):
        """Reset the agent's state for a new run."""
        self.t = 0 