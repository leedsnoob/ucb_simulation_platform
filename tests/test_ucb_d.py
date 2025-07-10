import sys
import os
import numpy as np
import pytest

from bandits.ucb_d import UCB_D

def test_ucb_d_initialization():
    """Tests if UCB_D initializes correctly."""
    bandit = UCB_D(n_arms=5, gamma=0.99)
    assert bandit.n_arms == 5
    assert bandit.gamma == 0.99
    assert np.array_equal(bandit.discounted_successes, np.zeros(5))
    assert np.array_equal(bandit.discounted_failures, np.zeros(5))

def test_ucb_d_update_logic():
    """Tests the core discounting update logic."""
    bandit = UCB_D(n_arms=3, gamma=0.9)
    
    # 1. First update for arm 0 (reward=1)
    bandit.update(arm=0, reward=1)
    assert bandit.discounted_successes[0] == 1
    assert bandit.discounted_failures[0] == 0
    assert bandit.discounted_successes[1] == 0 # Other arms unchanged

    # 2. Second update for arm 0 (reward=0)
    bandit.update(arm=0, reward=0)
    # Expected: successes=1*0.9=0.9, failures=0*0.9+1=1
    assert np.isclose(bandit.discounted_successes[0], 0.9)
    assert np.isclose(bandit.discounted_failures[0], 1.0)

    # 3. Update for arm 1 (reward=1)
    bandit.update(arm=1, reward=1)
    # Expected for arm 0: successes=0.9*0.9=0.81, failures=1.0*0.9=0.9
    # Expected for arm 1: successes=0*0.9+1=1, failures=0*0.9=0
    assert np.isclose(bandit.discounted_successes[0], 0.81)
    assert np.isclose(bandit.discounted_failures[0], 0.9)
    assert np.isclose(bandit.discounted_successes[1], 1.0)
    assert np.isclose(bandit.discounted_failures[1], 0.0)

def test_ucb_d_select_arm():
    """Tests the arm selection logic."""
    bandit = UCB_D(n_arms=3, gamma=0.9, alpha=0) # alpha=0 to remove exploration bonus for simplicity

    # Initial round-robin
    assert bandit.select_arm() == 0
    bandit.update(0, 1)
    assert bandit.select_arm() == 1
    bandit.update(1, 0)
    assert bandit.select_arm() == 2
    bandit.update(2, 0)

    # After initialization, arm 0 has a higher estimated value (1.0)
    # bandit.values should be approx [1.0, 0.0, 0.0]
    assert bandit.select_arm() == 0
    
    # Update arm 1 to have a better value
    bandit.update(1, 1)
    bandit.update(1, 1)
    bandit.update(1, 1)
    # Now arm 1 should have the highest value
    assert bandit.select_arm() == 1

def test_reset():
    """Tests if the reset method works correctly."""
    bandit = UCB_D(n_arms=3, gamma=0.9)
    bandit.update(0, 1)
    bandit.update(1, 0)
    
    bandit.reset()
    
    assert bandit.t == 0
    assert np.array_equal(bandit.discounted_successes, np.zeros(3))
    assert np.array_equal(bandit.discounted_failures, np.zeros(3)) 