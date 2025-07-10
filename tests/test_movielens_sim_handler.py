import unittest
import sys
import os

# Adjust path to import the handler from the parent directory's 'data' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.movielens_sim_handler import MovieLensSimHandler

class TestMovieLensSimHandler(unittest.TestCase):
    """
    Unit tests for the MovieLensSimHandler class.
    
    Note: This is an integration test as it relies on the actual data files.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the handler once for all tests to avoid repeated expensive I/O.
        We use a smaller K for faster testing.
        """
        print("\n--- Setting up TestMovieLensSimHandler ---")
        cls.n_clusters = 5  # Use a smaller K for faster tests
        cls.handler = MovieLensSimHandler(n_clusters=cls.n_clusters, data_dir='ml-1m')

    def test_initialization_and_arm_count(self):
        """Test if the handler initializes correctly and has the right number of arms."""
        print("Testing initialization and arm count...")
        self.assertIsNotNone(self.handler)
        self.assertEqual(self.handler.n_arms, self.n_clusters)

    def test_reward_pools_creation(self):
        """Test if reward pools are created for all arms."""
        print("Testing reward pools creation...")
        self.assertEqual(len(self.handler.reward_pools), self.n_clusters)
        # Check that all pools are non-empty lists of numbers
        for arm_id in range(self.n_clusters):
            self.assertIn(arm_id, self.handler.reward_pools)
            self.assertIsInstance(self.handler.reward_pools[arm_id], list)
            # It's possible, though unlikely, for a cluster to have no ratings.
            # We'll just check that the key exists.
            
    def test_true_mus_calculation(self):
        """Test the calculation of true mean rewards and mu_star."""
        print("Testing true mus calculation...")
        self.assertEqual(len(self.handler.true_mus), self.n_clusters)
        
        mu_star_manual = max(self.handler.true_mus.values())
        self.assertAlmostEqual(self.handler.mu_star, mu_star_manual, places=6)
        
        # Check that all mus are within the expected range of ratings (1-5)
        for mu in self.handler.true_mus.values():
            self.assertTrue(1 <= mu <= 5)

    def test_reward_sampling(self):
        """Test the reward sampling mechanism."""
        print("Testing reward sampling...")
        for arm_id in range(self.n_clusters):
            # Sample a few times to be reasonably sure
            for _ in range(10):
                reward = self.handler.sample_reward(arm_id)
                self.assertIsInstance(reward, (int, float))
                # Ratings are from 1 to 5
                self.assertTrue(1 <= reward <= 5)

if __name__ == '__main__':
    unittest.main() 