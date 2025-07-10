import unittest
import sys
import os
import numpy as np

# Adjust path to import the handler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.obd_sim_handler import OBDSimHandler

class TestOBDSimHandler(unittest.TestCase):
    """
    Unit tests for the OBDSimHandler class.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test data and handler once for all tests."""
        # Assuming 'obd' is in the parent directory of 'ucb_simulation_platform'
        # To make this test runnable from the project root, we construct the path
        current_dir = os.getcwd()
        # Navigate up to the project root if the test is run from the 'tests' directory
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, 'obd/random')
        print("\n--- Setting up TestOBDSimHandler ---")
        cls.handler = OBDSimHandler(data_dir=data_dir)
        cls.n_arms = 80 # We know this from prior exploration, let's assert it.

    def test_initialization_and_arm_count(self):
        """Test if the handler initializes correctly and has the right number of arms."""
        print("Testing initialization and arm count...")
        self.assertIsNotNone(self.handler)
        self.assertEqual(self.handler.n_arms, self.n_arms)
        self.assertEqual(len(self.handler.arm_ids), self.n_arms)

    def test_reward_pools_creation(self):
        """Test if reward pools are created for all arms."""
        print("Testing reward pools creation...")
        self.assertEqual(len(self.handler.reward_pools), self.n_arms)
        for arm_id in range(self.n_arms):
            self.assertIn(arm_id, self.handler.reward_pools)
            self.assertIsInstance(self.handler.reward_pools[arm_id], list)
            
    def test_true_mus_calculation(self):
        """Test the calculation of true mean rewards and mu_star."""
        print("Testing true mus calculation...")
        self.assertEqual(len(self.handler.true_mus), self.n_arms)
        
        mu_star_manual = max(self.handler.true_mus.values())
        self.assertAlmostEqual(self.handler.mu_star, mu_star_manual, places=6)
        
        # Check that all mus are within the expected range of rewards (0-1 for OBD)
        for mu in self.handler.true_mus.values():
            self.assertTrue(0 <= mu <= 1)

    def test_reward_sampling(self):
        """Test the reward sampling mechanism."""
        print("Testing reward sampling...")
        for arm_id in range(self.n_arms):
            # Sample a few times to be reasonably sure
            if self.handler.reward_pools[arm_id]: # Only test sampling if pool is not empty
                for _ in range(10):
                    reward = self.handler.sample_reward(arm_id)
                    self.assertIn(reward, [0, 1])

if __name__ == '__main__':
    unittest.main() 