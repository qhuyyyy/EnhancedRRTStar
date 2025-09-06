"""
Base RRT class providing common functionality for all RRT variants.

This module defines the abstract base class that all RRT algorithms inherit from,
providing common methods for tree management, sampling, and path planning.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any, Set
from dataclasses import dataclass
import math


@dataclass
class Node:
    """Represents a node in the RRT tree."""
    config: Tuple[float, float]
    parent: Optional['Node'] = None
    cost: float = 0.0
    children: List['Node'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def add_child(self, child: 'Node'):
        """Add a child node to this node."""
        self.children.append(child)
        child.parent = self
    
    def remove_child(self, child: 'Node'):
        """Remove a child node from this node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None


class BaseRRT(ABC):
    """
    Abstract base class for all RRT algorithms.
    
    This class provides common functionality including:
    - Tree management (add/remove nodes, find nearest neighbors)
    - Basic sampling strategies
    - Path extraction and validation
    - Common utility methods
    """
    
    def __init__(self, 
                 environment,
                 max_iterations: int = 5000,
                 step_size: float = 2.0,  # Updated to match paper
                 goal_bias: float = 0.05,
                 goal_tolerance: float = 1.0,  # Updated to match paper
                 random_seed: Optional[int] = None):
        """
        Initialize the base RRT planner.
        
        Args:
            environment: Environment object containing obstacles and bounds
            max_iterations: Maximum number of iterations for planning
            step_size: Maximum step size for tree extension
            goal_bias: Probability of sampling the goal configuration
            goal_tolerance: Distance threshold for considering goal reached
            random_seed: Random seed for reproducible behavior
        """
        self.environment = environment
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        
        # Tree structure
        self.nodes: List[Node] = []
        self.start_node: Optional[Node] = None
        self.goal_node: Optional[Node] = None
        
        # Planning state
        self.iteration = 0
        self.path_found = False
        self.final_path: List[Tuple[float, float]] = []
        
        # Statistics
        self.stats = {
            "iterations": 0,
            "nodes_created": 0,
            "collision_checks": 0,
            "planning_time": 0.0
        }
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def plan(self, 
             start: Tuple[float, float], 
             goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start configuration (x, y)
            goal: Goal configuration (x, y)
            
        Returns:
            List of waypoints defining the path from start to goal
        """
        import time
        start_time = time.time()
        
        # Initialize planning
        self._initialize_planning(start, goal)
        
        # Main planning loop
        while (self.iteration < self.max_iterations and 
               not self.path_found):
            
            self._planning_iteration()
            self.iteration += 1
        
        # Extract final path
        if self.path_found:
            self.final_path = self._extract_path()
        else:
            self.final_path = []
        
        # Update statistics
        self.stats["iterations"] = self.iteration
        self.stats["planning_time"] = time.time() - start_time
        
        return self.final_path
    
    def _initialize_planning(self, start: Tuple[float, float], goal: Tuple[float, float]):
        """Initialize the planning process."""
        # Validate start and goal configurations
        if not self.environment._is_valid_config(start):
            raise ValueError(f"Start configuration {start} is invalid")
        if not self.environment._is_valid_config(goal):
            raise ValueError(f"Goal configuration {goal} is invalid")
        
        # Set start and goal in environment
        self.environment.set_start_goal(start, goal)
        
        # Create start node
        self.start_node = Node(config=start, cost=0.0)
        self.nodes = [self.start_node]
        
        # Reset planning state
        self.iteration = 0
        self.path_found = False
        self.final_path = []
        
        # Reset statistics
        self.stats["nodes_created"] = 1
        self.stats["collision_checks"] = 0
    
    @abstractmethod
    def _planning_iteration(self):
        """Execute one planning iteration. Must be implemented by subclasses."""
        pass
    
    def _sample_configuration(self) -> Tuple[float, float]:
        """
        Sample a random configuration.
        
        Returns:
            Random configuration (x, y) in the configuration space
        """
        # Goal bias sampling
        if np.random.random() < self.goal_bias:
            return self.environment.goal_config
        
        # Uniform random sampling
        width, height = self.environment.dimensions
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        
        return (x, y)
    
    def _find_nearest_neighbor(self, config: Tuple[float, float]) -> Node:
        """
        Find the nearest neighbor to a given configuration.
        
        Args:
            config: Configuration to find nearest neighbor for
            
        Returns:
            Node that is closest to the given configuration
        """
        if not self.nodes:
            raise ValueError("No nodes in tree")
        
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.nodes:
            distance = self._distance(config, node.config)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _steer(self, 
                from_config: Tuple[float, float], 
                to_config: Tuple[float, float]) -> Tuple[float, float]:
        """
        Steer from one configuration toward another.
        
        Args:
            from_config: Starting configuration
            to_config: Target configuration
            
        Returns:
            New configuration that is at most step_size away from from_config
        """
        from_array = np.array(from_config)
        to_array = np.array(to_config)
        
        direction = to_array - from_array
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_config
        
        # Normalize and scale
        direction_normalized = direction / distance
        new_config = from_array + self.step_size * direction_normalized
        
        return tuple(new_config)
    
    def _is_collision_free(self, config: Tuple[float, float]) -> bool:
        """
        Check if a configuration is collision-free.
        
        Args:
            config: Configuration to check
            
        Returns:
            True if collision-free, False otherwise
        """
        self.stats["collision_checks"] += 1
        return self.environment._is_valid_config(config)
    
    def _is_path_collision_free(self, 
                                from_config: Tuple[float, float], 
                                to_config: Tuple[float, float]) -> bool:
        """
        Check if a path between two configurations is collision-free.
        
        Args:
            from_config: Starting configuration
            to_config: Ending configuration
            
        Returns:
            True if path is collision-free, False otherwise
        """
        return self.environment._is_line_segment_collision_free(
            from_config, to_config, self.step_size / 10
        )
    
    def _distance(self, config1: Tuple[float, float], config2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two configurations."""
        return np.linalg.norm(np.array(config1) - np.array(config2))
    
    def _add_node(self, config: Tuple[float, float], parent: Node, cost: float = 0.0) -> Node:
        """
        Add a new node to the tree.
        
        Args:
            config: Configuration for the new node
            parent: Parent node
            cost: Cost to reach this node
            
        Returns:
            The newly created node
        """
        new_node = Node(config=config, parent=parent, cost=cost)
        parent.add_child(new_node)
        self.nodes.append(new_node)
        self.stats["nodes_created"] += 1
        
        return new_node
    
    def _extract_path(self) -> List[Tuple[float, float]]:
        """
        Extract the path from start to goal.
        
        Returns:
            List of waypoints from start to goal
        """
        if not self.path_found or self.goal_node is None:
            return []
        
        path = []
        current_node = self.goal_node
        
        # Traverse from goal to start
        while current_node is not None:
            path.append(current_node.config)
            current_node = current_node.parent
        
        # Reverse to get start to goal order
        path.reverse()
        
        return path
    
    def _is_goal_reached(self, config: Tuple[float, float]) -> bool:
        """Check if the goal configuration has been reached."""
        return self._distance(config, self.environment.goal_config) <= self.goal_tolerance
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics."""
        return self.stats.copy()
    
    def get_tree_size(self) -> int:
        """Get the number of nodes in the tree."""
        return len(self.nodes)
    
    def reset(self):
        """Reset the planner to initial state."""
        self.nodes = []
        self.start_node = None
        self.goal_node = None
        self.iteration = 0
        self.path_found = False
        self.final_path = []
        self.stats = {
            "iterations": 0,
            "nodes_created": 0,
            "collision_checks": 0,
            "planning_time": 0.0
        } 