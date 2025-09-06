"""
Tests for RRT algorithm implementations.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environments.environment import Environment
from algorithms import StandardRRT, StandardRRTStar, InformedRRTStar, EnhancedRRT


class TestRRTAlgorithms(unittest.TestCase):
    """Test cases for RRT algorithm implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = Environment(dimensions=(15, 10), obstacle_density=0.2, random_seed=42)
        self.start = (1, 1)
        self.goal = (13, 8)
        self.env.set_start_goal(self.start, self.goal)
    
    def test_standard_rrt(self):
        """Test Standard RRT algorithm."""
        planner = StandardRRT(self.env, max_iterations=1000, random_seed=42)
        path = planner.plan(self.start, self.goal)
        
        # Should find a path
        self.assertGreater(len(path), 0)
        
        # Path should start and end correctly (with tolerance for goal_tolerance)
        self.assertAlmostEqual(path[0][0], self.start[0], places=1)
        self.assertAlmostEqual(path[0][1], self.start[1], places=1)
        # Goal tolerance is 1.0, so check within that range
        goal_distance = np.linalg.norm(np.array(path[-1]) - np.array(self.goal))
        self.assertLessEqual(goal_distance, 1.0)
        
        # Should have reasonable tree size
        self.assertGreater(len(planner.nodes), 10)
    
    def test_standard_rrt_star(self):
        """Test Standard RRT* algorithm."""
        planner = StandardRRTStar(self.env, max_iterations=1000, random_seed=42)
        path = planner.plan(self.start, self.goal)
        
        # Should find a path
        self.assertGreater(len(path), 0)
        
        # Path should start and end correctly (with tolerance for goal_tolerance)
        self.assertAlmostEqual(path[0][0], self.start[0], places=1)
        self.assertAlmostEqual(path[0][1], self.start[1], places=1)
        # Goal tolerance is 1.0, so check within that range
        goal_distance = np.linalg.norm(np.array(path[-1]) - np.array(self.goal))
        self.assertLessEqual(goal_distance, 1.0)
        
        # Should have reasonable tree size
        self.assertGreater(len(planner.nodes), 10)
    
    def test_informed_rrt_star(self):
        """Test Informed RRT* algorithm."""
        planner = InformedRRTStar(self.env, max_iterations=1000, random_seed=42)
        path = planner.plan(self.start, self.goal)
        
        # Should find a path
        self.assertGreater(len(path), 0)
        
        # Path should start and end correctly (with tolerance for goal_tolerance)
        self.assertAlmostEqual(path[0][0], self.start[0], places=1)
        self.assertAlmostEqual(path[0][1], self.start[1], places=1)
        # Goal tolerance is 1.0, so check within that range
        goal_distance = np.linalg.norm(np.array(path[-1]) - np.array(self.goal))
        self.assertLessEqual(goal_distance, 1.0)
        
        # Should have reasonable tree size
        self.assertGreater(len(planner.nodes), 10)
    
    def test_enhanced_rrt(self):
        """Test Enhanced RRT* algorithm."""
        planner = EnhancedRRT(self.env, max_iterations=1000, random_seed=42)
        path = planner.plan(self.start, self.goal)
        
        # Should find a path
        self.assertGreater(len(path), 0)
        
        # Path should start and end correctly (with tolerance for goal_tolerance)
        self.assertAlmostEqual(path[0][0], self.start[0], places=1)
        self.assertAlmostEqual(path[0][1], self.start[1], places=1)
        # Goal tolerance is 1.0, so check within that range
        goal_distance = np.linalg.norm(np.array(path[-1]) - np.array(self.goal))
        self.assertLessEqual(goal_distance, 1.0)
        
        # Should have reasonable tree size
        self.assertGreater(len(planner.nodes), 10)
        
        # Test enhanced statistics
        stats = planner.get_enhanced_stats()
        self.assertIn('adaptive_goal_bias', stats)
        self.assertIn('adaptive_sampling_factor', stats)
        self.assertIn('sampling_bias_coefficient', stats)
    
    def test_path_accuracy_calculation(self):
        """Test path accuracy calculation."""
        from experiments.experiment_runner import ExperimentRunner
        
        runner = ExperimentRunner()
        
        # Test with a straight line path
        straight_path = [(0, 0), (10, 0)]
        accuracy = runner._calculate_path_accuracy(straight_path, (0, 0), (10, 0))
        self.assertAlmostEqual(accuracy, 1.0, places=2)  # Should be optimal
        
        # Test with a longer path
        longer_path = [(0, 0), (5, 0), (10, 0)]
        accuracy = runner._calculate_path_accuracy(longer_path, (0, 0), (10, 0))
        self.assertLessEqual(accuracy, 1.0)  # Should be less than or equal to optimal
        
        # Test with empty path
        accuracy = runner._calculate_path_accuracy([], (0, 0), (10, 0))
        self.assertEqual(accuracy, 0.0)
    
    def test_algorithm_comparison(self):
        """Test that all algorithms can be compared fairly."""
        algorithms = [
            StandardRRT(self.env, max_iterations=500, random_seed=42),
            StandardRRTStar(self.env, max_iterations=500, random_seed=42),
            InformedRRTStar(self.env, max_iterations=500, random_seed=42),
            EnhancedRRT(self.env, max_iterations=500, random_seed=42)
        ]
        
        results = []
        for planner in algorithms:
            path = planner.plan(self.start, self.goal)
            results.append({
                'success': len(path) > 0,
                'path_length': len(path),
                'tree_size': len(planner.nodes),
                'iterations': planner.iteration
            })
        
        # All algorithms should find paths
        for result in results:
            self.assertTrue(result['success'])
        
        # All should have reasonable performance
        for result in results:
            self.assertGreater(result['path_length'], 0)
            self.assertGreater(result['tree_size'], 0)
            self.assertGreater(result['iterations'], 0)


if __name__ == '__main__':
    unittest.main()
