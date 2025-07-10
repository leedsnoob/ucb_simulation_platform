from .base_bandit import BaseBandit
import numpy as np
import collections

class UCB_D(BaseBandit):
    """
    Discounted Upper Confidence Bound (UCB) algorithm.
    It correctly discounts past observations by decaying the counts of
    successes and failures, adapting to non-stationary environments.
    """
    def __init__(self, n_arms: int, alpha: float = 1.0, gamma: float = 0.99, **kwargs):
        """
        Initializes the Discounted UCB algorithm.

        Parameters
        ----------
        n_arms : int
            Number of arms.
        alpha : float
            Exploration parameter.
        gamma : float
            Discounting factor, between 0 and 1.
        """
        super().__init__(n_arms=n_arms, alpha=alpha, **kwargs)
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

        # Discounted statistics for successes and failures (like alpha and beta in Beta dist)
        self.discounted_successes = np.zeros(n_arms)
        self.discounted_failures = np.zeros(n_arms)

    def select_arm(self, context=None) -> int:
        """Selects an arm according to the Discounted UCB policy."""
        total_discounted_counts = self.discounted_successes + self.discounted_failures
        
        # Play each arm once to initialize
        for arm in range(self.n_arms):
            if total_discounted_counts[arm] == 0:
                return arm
        
        # Calculate UCB values based on discounted stats
        ucb_values = np.zeros(self.n_arms)
        total_counts = np.sum(total_discounted_counts)

        for arm in range(self.n_arms):
            arm_total_discounted_counts = total_discounted_counts[arm]
            if arm_total_discounted_counts == 0: # Should not happen after init
                ucb_values[arm] = float('inf')
                continue

            average_reward = self.discounted_successes[arm] / arm_total_discounted_counts
            bonus = self.alpha * np.sqrt(np.log(total_counts) / arm_total_discounted_counts)
            ucb_values[arm] = average_reward + bonus
            
        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float) -> None:
        """Updates the discounted statistics for all arms."""
        # Discount all arms first
        self.discounted_successes *= self.gamma
        self.discounted_failures *= self.gamma

        # Update the selected arm
        if reward == 1:
            self.discounted_successes[arm] += 1
        else:
            self.discounted_failures[arm] += 1
        
        self.t += 1 # Base class t for total steps if needed

    def reset(self) -> None:
        """Resets the agent to its initial state."""
        super().reset()
        self.discounted_successes.fill(0)
        self.discounted_failures.fill(0)

    @property
    def values(self) -> np.ndarray:
        """Returns the current estimated values of the arms."""
        total_counts = self.discounted_successes + self.discounted_failures
        # To avoid division by zero, add a small epsilon
        return self.discounted_successes / (total_counts + 1e-8)

    @property
    def counts(self) -> np.ndarray:
        """Returns the effective (discounted) counts of the arms."""
        return self.discounted_successes + self.discounted_failures

class UCB_SW(BaseBandit):
    """
    Sliding Window Upper Confidence Bound (SW-UCB) algorithm.
    It only considers rewards within a fixed-size window.
    """
    def __init__(self, n_arms: int, window_size: int, c: float = 2.0):
        super().__init__(n_arms)
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.c = c
        self.rewards_window = {i: collections.deque(maxlen=window_size) for i in range(n_arms)}

    def select_arm(self):
        """Selects an arm based on UCB calculated from the sliding window."""
        for arm in range(self.n_arms):
            if not self.rewards_window[arm]:
                return arm

        total_counts = sum(len(self.rewards_window[arm]) for arm in range(self.n_arms))
        if total_counts == 0:
             return np.random.randint(self.n_arms)

        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if not self.rewards_window[arm]:
                ucb_values[arm] = float('inf') # Prioritize unexplored arms
                continue
            
            window = self.rewards_window[arm]
            mean_reward = np.mean(window)
            bonus = self.c * np.sqrt(np.log(total_counts) / len(window))
            ucb_values[arm] = mean_reward + bonus
            
        return np.argmax(ucb_values)

    def update(self, arm_idx: int, reward: float):
        """Updates the sliding window for the chosen arm."""
        self.rewards_window[arm_idx].append(reward)
        self.t += 1

class FDSW_UCB(BaseBandit):
    """
    f-Discounted Sliding Window UCB algorithm, inspired by the f-dsw TS paper.
    This algorithm maintains two estimates for each arm:
    1. A discounted estimate over the entire history.
    2. A sliding window estimate over recent rewards.
    It uses an aggregation function f to combine these two estimates for decision making.
    """
    def __init__(self, n_arms: int, gamma: float, window_size: int, agg_function: str = 'mean', c: float = 2.0):
        super().__init__(n_arms)
        if agg_function not in ['min', 'max', 'mean']:
            raise ValueError("agg_function must be 'min', 'max', or 'mean'.")

        # Parameters
        self.gamma = gamma
        self.window_size = window_size
        self.agg_function = agg_function
        self.c = c
        
        # For Discounted UCB part
        self.discounted_rewards_sum = np.zeros(n_arms)
        self.discounted_counts = np.zeros(n_arms)
        
        # For Sliding Window UCB part
        self.rewards_window = {i: collections.deque(maxlen=window_size) for i in range(n_arms)}
    
    def select_arm(self):
        """Selects an arm by combining discounted and sliding-window UCB scores."""
        # Initial exploration phase
        for arm in range(self.n_arms):
            if self.discounted_counts[arm] == 0:
                return arm
                
        # --- Calculate UCB for both views ---
        ucb_d = np.zeros(self.n_arms) # UCB from discounted history
        ucb_sw = np.zeros(self.n_arms) # UCB from sliding window
        
        total_discounted_counts = np.sum(self.discounted_counts)
        total_window_counts = sum(len(self.rewards_window[arm]) for arm in range(self.n_arms))

        for arm in range(self.n_arms):
            # Discounted UCB
            mean_d = self.discounted_rewards_sum[arm] / self.discounted_counts[arm]
            bonus_d = self.c * np.sqrt(np.log(total_discounted_counts) / self.discounted_counts[arm])
            ucb_d[arm] = mean_d + bonus_d

            # Sliding Window UCB
            if not self.rewards_window[arm]:
                 ucb_sw[arm] = float('inf') # Should not happen after initial exploration
            else:
                window = self.rewards_window[arm]
                mean_sw = np.mean(window)
                bonus_sw = self.c * np.sqrt(np.log(total_window_counts) / len(window))
                ucb_sw[arm] = mean_sw + bonus_sw
        
        # --- Aggregation ---
        if self.agg_function == 'min':
            final_ucb = np.minimum(ucb_d, ucb_sw)
        elif self.agg_function == 'max':
            final_ucb = np.maximum(ucb_d, ucb_sw)
        else: # mean
            final_ucb = (ucb_d + ucb_sw) / 2.0
            
        return np.argmax(final_ucb)

    def update(self, arm_idx: int, reward: float):
        """Updates both the discounted history and the sliding window."""
        self.t += 1
        
        # 1. Update Discounted history
        self.discounted_rewards_sum *= self.gamma
        self.discounted_counts *= self.gamma
        self.discounted_rewards_sum[arm_idx] += reward
        self.discounted_counts[arm_idx] += 1
        
        # 2. Update Sliding Window
        self.rewards_window[arm_idx].append(reward)

# Keep original UCB_DSW for compatibility if needed, but mark as deprecated.
class UCB_DSW(BaseBandit):
    """
    DEPRECATED: This is a custom, non-standard implementation. 
    Use FDSW_UCB for a paper-compliant algorithm.
    """
    def __init__(self, n_arms, gamma, window_size, c=2):
        super().__init__(n_arms)
        self.gamma = gamma
        self.window_size = window_size
        self.c = c
        self.rewards_window = {i: collections.deque(maxlen=window_size) for i in range(n_arms)}
        self.discounted_counts = np.zeros(n_arms)

    def select_arm(self):
        for i in range(self.n_arms):
            if self.arm_counts[i] == 0:
                return i
        
        total_counts = np.sum(self.arm_counts)
        ucb_values = np.zeros(self.n_arms)
        
        for i in range(self.n_arms):
            bonus = self.c * np.sqrt(np.log(total_counts) / self.arm_counts[i])
            ucb_values[i] = self.estimated_rewards[i] + bonus

        return np.argmax(ucb_values)

    def update(self, arm_idx, reward):
        self.rewards_window[arm_idx].append(reward)
        self.arm_counts[arm_idx] += 1
        self.t += 1
        
        # Update discounted counts
        self.discounted_counts *= self.gamma
        self.discounted_counts[arm_idx] += 1

        # Update estimated rewards using a mix of sliding window and discounted counts
        window = self.rewards_window[arm_idx]
        if window:
            mean_sw_reward = sum(window) / len(window)
            
            # Custom update rule
            discount_factor = self.discounted_counts[arm_idx] / sum(self.discounted_counts)
            
            # This is a non-standard heuristic combination
            self.estimated_rewards[arm_idx] = (1 - discount_factor) * self.estimated_rewards[arm_idx] + discount_factor * mean_sw_reward 