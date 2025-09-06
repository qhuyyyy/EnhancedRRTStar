"""
Tests for the Environment class.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environments.environment import Environment, Obstacle


class TestEnvironment(unittest.TestCase):
    """Test cases for Environment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = Environment(dimensions=(10, 10), obstacle_density=0.2, random_seed=42)
    
    def test_environment_creation(self):
        """Test environment creation with specified parameters."""
        self.assertEqual(self.env.dimensions, (10, 10))
        self.assertEqual(self.env.obstacle_density, 0.2)
        self.assertIsInstance(self.env.obstacles, list)
    
    def test_obstacle_creation(self):
        """Test that obstacles are created and don't overlap."""
        # Check that obstacles exist
        self.assertGreater(len(self.env.obstacles), 0)
        
        # Check that obstacles don't overlap
        for i, obs1 in enumerate(self.env.obstacles):
            for j, obs2 in enumerate(self.env.obstacles):
                if i != j:
                    # Obstacles should not overlap
                    self.assertFalse(self.env._rectangles_overlap(obs1, obs2))
    
    def test_start_goal_setting(self):
        """Test setting start and goal configurations."""
        start = (1, 1)
        goal = (8, 8)
        
        self.env.set_start_goal(start, goal)
        self.assertEqual(self.env.start_config, start)
        self.assertEqual(self.env.goal_config, goal)
    
    def test_invalid_config_validation(self):
        """Test validation of invalid configurations."""
        # Test out of bounds
        invalid_configs = [
            (-1, 5),  # Negative x
            (5, -1),  # Negative y
            (11, 5),  # Beyond width
            (5, 11),  # Beyond height
        ]
        
        for config in invalid_configs:
            self.assertFalse(self.env._is_valid_config(config))
    
    def test_collision_detection(self):
        """Test collision detection between obstacles and configurations."""
        # Create a known obstacle
        obstacle = Obstacle(5, 5, 2, 2)
        
        # Test collision detection
        self.assertTrue(obstacle.contains((5.5, 5.5)))  # Inside
        self.assertFalse(obstacle.contains((4, 4)))      # Outside
        self.assertTrue(obstacle.contains((5, 5)))      # On edge
        self.assertTrue(obstacle.contains((7, 7)))      # On edge
    
    def test_path_collision_detection(self):
        """Test path collision detection."""
        # Create environment without obstacles for this test
        env_no_obstacles = Environment(dimensions=(10, 10), obstacle_density=0.0, random_seed=42)
        
        # Create a simple path
        path = [(1, 1), (9, 9)]
        
        # Should be collision-free in empty environment
        self.assertTrue(env_no_obstacles.is_path_collision_free(path))
    
    def test_environment_stats(self):
        """Test environment statistics calculation."""
        stats = self.env.get_environment_stats()
        
        self.assertIn('dimensions', stats)
        self.assertIn('total_area', stats)
        self.assertIn('num_obstacles', stats)
        self.assertIn('actual_density', stats)
        
        # Check that actual density is reasonable
        self.assertGreaterEqual(stats['actual_density'], 0)
        self.assertLessEqual(stats['actual_density'], 1)


class TestObstacle(unittest.TestCase):
    """Test cases for Obstacle class."""
    
    def test_obstacle_creation(self):
        """Test obstacle creation and properties."""
        obstacle = Obstacle(1, 2, 3, 4)
        
        self.assertEqual(obstacle.x, 1)
        self.assertEqual(obstacle.y, 2)
        self.assertEqual(obstacle.width, 3)
        self.assertEqual(obstacle.height, 4)
    
    def test_obstacle_bounds(self):
        """Test obstacle bounds calculation."""
        obstacle = Obstacle(1, 2, 3, 4)
        bounds = obstacle.get_bounds()
        
        self.assertEqual(bounds, (1, 2, 4, 6))  # (x_min, y_min, x_max, y_max)
    
    def test_obstacle_contains(self):
        """Test point containment in obstacles."""
        obstacle = Obstacle(1, 2, 3, 4)
        
        # Test points inside
        self.assertTrue(obstacle.contains((2, 3)))
        self.assertTrue(obstacle.contains((1, 2)))  # On edge
        self.assertTrue(obstacle.contains((4, 6)))  # On edge
        
        # Test points outside
        self.assertFalse(obstacle.contains((0, 3)))
        self.assertFalse(obstacle.contains((2, 1)))
        self.assertFalse(obstacle.contains((5, 3)))


if __name__ == '__main__':
    unittest.main() 