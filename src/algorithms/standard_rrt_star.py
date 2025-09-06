"""
Standard RRT* implementation providing asymptotic optimality.

This module implements the RRT* algorithm as described in the original
paper by Karaman and Frazzoli, which provides asymptotic optimality
through systematic rewiring operations.
"""

import numpy as np
from typing import Tuple, List, Optional, Set
from .base_rrt import BaseRRT, Node


class StandardRRTStar(BaseRRT):
    """
    Standard RRT* algorithm implementation.
    
    This algorithm extends RRT with:
    - Rewiring operations for asymptotic optimality
    - Cost-based parent selection
    - Neighbor-based tree optimization
    """
    
    def __init__(self, 
                 environment,
                 max_iterations: int = 5000,
                 step_size: float = 2.0,  # Updated to match paper
                 goal_bias: float = 0.05,
                 goal_tolerance: float = 1.0,  # Updated to match paper
                 rewiring_radius: float = 2.0,
                 random_seed: Optional[int] = None):
        """
        Initialize Standard RRT* planner.
        
        Args:
            environment: Environment object containing obstacles and bounds
            max_iterations: Maximum number of iterations for planning
            step_size: Maximum step size for tree extension
            goal_bias: Probability of sampling the goal configuration
            goal_tolerance: Distance threshold for considering goal reached
            rewiring_radius: Radius for finding neighbors during rewiring
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
        self.rewiring_radius = rewiring_radius
    
    def _planning_iteration(self):
        """
        Execute one planning iteration of Standard RRT*.
        
        The algorithm:
        1. Samples a random configuration
        2. Finds the nearest neighbor in the tree
        3. Extends the tree toward the sampled configuration
        4. Performs rewiring operations for optimality
        5. Checks if goal is reached
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
                # Find optimal parent for new node
                optimal_parent = self._find_optimal_parent(new_config, nearest_node)
                
                if optimal_parent is not None:
                    # Calculate cost through optimal parent
                    new_cost = optimal_parent.cost + self._distance(optimal_parent.config, new_config)
                    
                    # Add new node to tree
                    new_node = self._add_node(new_config, optimal_parent, new_cost)
                    
                    # Perform rewiring operations
                    self._rewire_tree(new_node)
                    
                    # Check if goal is reached
                    if self._is_goal_reached(new_config):
                        self.goal_node = new_node
                        self.path_found = True
                        return
    
    def _find_optimal_parent(self, 
                             new_config: Tuple[float, float], 
                             nearest_node: Node) -> Optional[Node]:
        """
        Find the optimal parent for a new configuration.
        
        Args:
            new_config: New configuration to find parent for
            nearest_node: Nearest neighbor node (initial candidate)
            
        Returns:
            Optimal parent node that minimizes cost to new configuration
        """
        # Start with nearest node as candidate
        best_parent = nearest_node
        best_cost = nearest_node.cost + self._distance(nearest_node.config, new_config)
        
        # Find all nodes within rewiring radius
        neighbors = self._find_neighbors(new_config, self.rewiring_radius)
        
        # Check each neighbor as potential parent
        for neighbor in neighbors:
            # Check if path through neighbor is collision-free
            if self._is_path_collision_free(neighbor.config, new_config):
                # Calculate cost through this neighbor
                cost_through_neighbor = neighbor.cost + self._distance(neighbor.config, new_config)
                
                # Update if this neighbor provides better cost
                if cost_through_neighbor < best_cost:
                    best_cost = cost_through_neighbor
                    best_parent = neighbor
        
        return best_parent
    
    def _find_neighbors(self, 
                        config: Tuple[float, float], 
                        radius: float) -> List[Node]:
        """
        Find all nodes within a given radius of a configuration.
        
        Args:
            config: Center configuration
            radius: Search radius
            
        Returns:
            List of nodes within the specified radius
        """
        neighbors = []
        for node in self.nodes:
            if self._distance(config, node.config) <= radius:
                neighbors.append(node)
        return neighbors
    
    def _rewire_tree(self, new_node: Node):
        """
        Perform rewiring operations to optimize the tree.
        
        Args:
            new_node: Newly added node to consider for rewiring
        """
        # Find neighbors of new node
        neighbors = self._find_neighbors(new_node.config, self.rewiring_radius)
        
        # Check if any neighbor can improve its cost through new node
        for neighbor in neighbors:
            if neighbor == new_node.parent:
                continue  # Skip parent
            
            # Check if path through new node is collision-free
            if self._is_path_collision_free(new_node.config, neighbor.config):
                # Calculate potential new cost for neighbor
                new_cost_for_neighbor = new_node.cost + self._distance(new_node.config, neighbor.config)
                
                # If new cost is better, rewire
                if new_cost_for_neighbor < neighbor.cost:
                    # Remove old parent connection
                    if neighbor.parent is not None:
                        neighbor.parent.remove_child(neighbor)
                    
                    # Update parent and cost
                    neighbor.parent = new_node
                    neighbor.cost = new_cost_for_neighbor
                    new_node.add_child(neighbor)
                    
                    # Propagate cost changes to children
                    self._propagate_cost_changes(neighbor)
    
    def _propagate_cost_changes(self, node: Node):
        """
        Propagate cost changes to all children of a node.
        
        Args:
            node: Node whose cost has changed
        """
        for child in node.children:
            # Update child cost
            old_cost = child.cost
            new_cost = node.cost + self._distance(node.config, child.config)
            child.cost = new_cost
            
            # Continue propagating to children
            self._propagate_cost_changes(child) 