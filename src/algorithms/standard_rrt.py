"""
Standard RRT (Rapidly-exploring Random Tree) implementation.

This module provides the basic RRT algorithm implementation as described
in the original paper by LaValle and Kuffner.
"""

import numpy as np
from typing import Tuple, List, Optional
from .base_rrt import BaseRRT, Node


class StandardRRT(BaseRRT):
    """
    Standard RRT algorithm implementation.
    
    This is the basic RRT algorithm that provides:
    - Random sampling in configuration space
    - Tree expansion toward sampled configurations
    - Basic goal bias sampling
    - No path optimization (suboptimal paths)
    """
    
    def __init__(self, 
                 environment,
                 max_iterations: int = 5000,
                 step_size: float = 1.0,
                 goal_bias: float = 0.05,
                 goal_tolerance: float = 0.5,
                 random_seed: Optional[int] = None):
        """
        Initialize Standard RRT planner.
        
        Args:
            environment: Environment object containing obstacles and bounds
            max_iterations: Maximum number of iterations for planning
            step_size: Maximum step size for tree extension
            goal_bias: Probability of sampling the goal configuration
            goal_tolerance: Distance threshold for considering goal reached
            random_seed: Random seed for reproducible behavior
        """
        super().__init__(
            environment=environment,
            max_iterations=max_iterations,
            step_size=step_size,
            goal_bias=goal_bias,
            goal_tolerance=goal_tolerance,
            random_seed=random_seed
        )
    
    def _planning_iteration(self):
        """
        Execute one planning iteration of Standard RRT.
        
        The algorithm:
        1. Samples a random configuration
        2. Finds the nearest neighbor in the tree
        3. Extends the tree toward the sampled configuration
        4. Checks if goal is reached
        """
        # Sample random configuration
        random_config = self._sample_configuration()
        
        # Find nearest neighbor
        nearest_node = self._find_nearest_neighbor(random_config)
        
        # Extend tree toward random configuration
        new_config = self._steer(nearest_node.config, random_config)
        
        # Check if new configuration is collision-free
        if self._is_collision_free(new_config):
            # Check if path to new configuration is collision-free
            if self._is_path_collision_free(nearest_node.config, new_config):
                # Calculate cost to new configuration
                new_cost = nearest_node.cost + self._distance(nearest_node.config, new_config)
                
                # Add new node to tree
                new_node = self._add_node(new_config, nearest_node, new_cost)
                
                # Check if goal is reached
                if self._is_goal_reached(new_config):
                    self.goal_node = new_node
                    self.path_found = True
                    return
        
        # If no valid extension, continue to next iteration
        pass 