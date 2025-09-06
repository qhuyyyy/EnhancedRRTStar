"""
Informed RRT* implementation with ellipsoidal sampling.

This module implements the Informed RRT* algorithm that restricts sampling
to an ellipsoidal region containing all configurations with cost lower
than the current best solution, dramatically improving convergence speed.
"""

import numpy as np
from typing import Tuple, List, Optional
from .standard_rrt_star import StandardRRTStar, Node


class InformedRRTStar(StandardRRTStar):
    """
    Informed RRT* algorithm implementation.
    
    This algorithm extends RRT* with:
    - Ellipsoidal sampling region based on current best solution
    - Focused exploration in promising areas
    - Improved convergence speed
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
        Initialize Informed RRT* planner.
        
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
            rewiring_radius=rewiring_radius,
            random_seed=random_seed
        )
        
        # Informed sampling parameters
        self.best_solution_cost = float('inf')
        self.sampling_ellipsoid = None
    
    def _sample_configuration(self) -> Tuple[float, float]:
        """
        Sample configuration using informed sampling when available.
        
        Returns:
            Sampled configuration (x, y) in the configuration space
        """
        # Goal bias sampling
        if np.random.random() < self.goal_bias:
            return self.environment.goal_config
        
        # Use informed sampling if we have a solution
        if self.best_solution_cost < float('inf'):
            return self._sample_informed_configuration()
        
        # Fall back to uniform sampling
        return super()._sample_configuration()
    
    def _sample_informed_configuration(self) -> Tuple[float, float]:
        """
        Sample configuration within the informed ellipsoidal region.
        
        Returns:
            Sampled configuration within the ellipsoidal sampling region
        """
        # Get start and goal configurations
        start = self.environment.start_config
        goal = self.environment.goal_config
        
        # Calculate ellipsoid parameters
        c_min = self._distance(start, goal)  # Minimum cost
        c_max = self.best_solution_cost       # Current best solution cost
        
        if c_max <= c_min:
            # If current solution is optimal, sample uniformly
            return super()._sample_configuration()
        
        # Calculate ellipsoid center and axes
        center = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
        
        # Transform to unit sphere
        start_array = np.array(start)
        goal_array = np.array(goal)
        center_array = np.array(center)
        
        # Calculate rotation matrix to align with x-axis
        direction = goal_array - start_array
        direction_normalized = direction / np.linalg.norm(direction)
        
        # Create rotation matrix (simplified for 2D)
        cos_theta = direction_normalized[0]
        sin_theta = direction_normalized[1]
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]])
        
        # Sample in unit sphere
        while True:
            # Sample uniform random point in unit sphere
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            
            # Convert to spherical coordinates
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            
            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            
            # Scale by ellipsoid parameters
            a = c_max / 2  # Semi-major axis
            b = np.sqrt(c_max**2 - c_min**2) / 2  # Semi-minor axis
            
            x_scaled = a * x
            y_scaled = b * y
            
            # Transform back to original coordinate system
            point = np.array([x_scaled, y_scaled])
            point_rotated = rotation_matrix @ point
            point_translated = point_rotated + center_array
            
            # Check bounds
            width, height = self.environment.dimensions
            if (0 <= point_translated[0] <= width and 
                0 <= point_translated[1] <= height):
                return tuple(point_translated)
    
    def _planning_iteration(self):
        """
        Execute one planning iteration of Informed RRT*.
        
        The algorithm extends Standard RRT* with informed sampling.
        """
        # Execute standard RRT* iteration
        super()._planning_iteration()
        
        # Update best solution cost if we found a path
        if self.path_found and self.goal_node is not None:
            current_cost = self.goal_node.cost
            if current_cost < self.best_solution_cost:
                self.best_solution_cost = current_cost
    
    def _initialize_planning(self, start: Tuple[float, float], goal: Tuple[float, float]):
        """Initialize the planning process with informed sampling setup."""
        super()._initialize_planning(start, goal)
        
        # Reset informed sampling parameters
        self.best_solution_cost = float('inf')
        self.sampling_ellipsoid = None
    
    def get_informed_sampling_stats(self) -> dict:
        """Get statistics about informed sampling usage."""
        return {
            "best_solution_cost": self.best_solution_cost,
            "has_informed_sampling": self.best_solution_cost < float('inf'),
            "sampling_region_type": "ellipsoidal" if self.best_solution_cost < float('inf') else "uniform"
        } 