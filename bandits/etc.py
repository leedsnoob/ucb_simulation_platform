import numpy as np
import random

class ETC:
    """
    Explore-Then-Commit (ETC) algorithm.
    
    It first explores each arm for a fixed number of steps and then
    commits to the best-performing arm for the remaining steps.
    """

    def __init__(self, n_arms: int, exploration_budget: int, **kwargs):
        """
        Initializes the Explore-Then-Commit algorithm.

        Parameters
        ----------
        n_arms : int
            Number of arms.
        exploration_budget : int
            The total number of steps dedicated to exploration. Each arm will be
            pulled exploration_budget / n_arms times.
        """
        self.n_arms = n_arms
        if exploration_budget <= 0:
            raise ValueError("exploration_budget must be a positive integer")
        
        self.exploration_budget = exploration_budget
        # Each arm is pulled at least this many times.
        # The budget might not be perfectly divisible.
        self.pulls_per_arm = self.exploration_budget // self.n_arms
        self.total_steps = 0
        self.best_arm = -1

        self._counts = np.zeros(n_arms)
        self._values = np.zeros(n_arms)

    def select_arm(self, context=None) -> int:
        """
        Selects an arm based on the current phase (exploration or exploitation).
        """
        if self.total_steps < self.exploration_budget:
            # Exploration Phase: pull each arm in a round-robin fashion
            arm_to_pull = int(self.total_steps % self.n_arms)
            return arm_to_pull
        else:
            # Exploitation Phase
            if self.best_arm == -1:
                # First step after exploration: determine the best arm
                self.best_arm = np.argmax(self._values)
            return self.best_arm

    def update(self, arm: int, reward: float) -> None:
        """Updates the statistics for the chosen arm."""
        self.total_steps += 1
        
        if self._counts[arm] < self.pulls_per_arm:
            # Update stats only during the initial pulls for each arm in the exploration phase
        self._counts[arm] += 1
            n = self._counts[arm]
            self._values[arm] = ((n - 1) * self._values[arm] + reward) / n

    def reset(self) -> None:
        """Resets the agent to its initial state."""
        self.total_steps = 0
        self.best_arm = -1
        self._counts.fill(0)
        self._values.fill(0)

    @property
    def values(self) -> np.ndarray:
        """Returns the current estimated values of the arms."""
        return self._values

    @property
    def counts(self) -> np.ndarray:
        """Returns the real counts of the arms."""
        return self._counts 