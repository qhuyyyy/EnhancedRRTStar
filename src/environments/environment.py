"""
Environment class for managing configuration space and obstacles.

This module provides the core environment functionality including:
- Configuration space management
- Obstacle representation
- Collision detection
- Start/goal configuration management
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Obstacle:
	"""Represents a rectangular obstacle in the environment."""
	x: float
	y: float
	width: float
	height: float
	
	def contains(self, point: Tuple[float, float]) -> bool:
		"""Check if a point is inside this obstacle."""
		px, py = point
		return (self.x <= px <= self.x + self.width and 
				self.y <= py <= self.y + self.height)
	
	def get_bounds(self) -> Tuple[float, float, float, float]:
		"""Get the bounds of the obstacle (x_min, y_min, x_max, y_max)."""
		return (self.x, self.y, self.x + self.width, self.y + self.height)


class Environment:
	"""
	Environment class for managing configuration space and obstacles.
	
	This class provides functionality for:
	- Creating environments with specified dimensions and obstacle density
	- Collision detection between paths and obstacles
	- Managing start and goal configurations
	- Environment validation and statistics
	"""
	
	def __init__(self, 
				 dimensions: Tuple[int, int] = (25, 20),
				 obstacle_density: float = 0.287,
				 random_seed: Optional[int] = None,
				 obstacles: Optional[List[Obstacle]] = None):
		"""
		Initialize the environment.
		
		Args:
			dimensions: Tuple of (width, height) for the environment
			obstacle_density: Density of obstacles (0.0 to 1.0)
			random_seed: Random seed for reproducible obstacle generation
			obstacles: Optional fixed list of obstacles; if provided, random generation is skipped
		"""
		self.dimensions = dimensions
		self.obstacle_density = obstacle_density
		self.obstacles: List[Obstacle] = []
		self.start_config: Optional[Tuple[float, float]] = None
		self.goal_config: Optional[Tuple[float, float]] = None
		
		if random_seed is not None:
			np.random.seed(random_seed)
		
		# Use fixed obstacles if provided
		if obstacles is not None and len(obstacles) > 0:
			self.obstacles = obstacles
		else:
			self._generate_obstacles()
	
	def _generate_obstacles(self):
		"""Generate obstacles based on the specified density and dimensions."""
		width, height = self.dimensions
		total_area = width * height
		target_obstacle_area = total_area * self.obstacle_density
		
		# Generate obstacles until we reach the target density
		current_obstacle_area = 0
		max_attempts = 1000
		attempts = 0
		
		while (current_obstacle_area < target_obstacle_area and 
			   attempts < max_attempts):
			
			# Random obstacle dimensions (1x1 to 4x4)
			obs_width = np.random.uniform(1, 4)
			obs_height = np.random.uniform(1, 4)
			
			# Random position
			x = np.random.uniform(0, width - obs_width)
			y = np.random.uniform(0, height - obs_height)
			
			# Create obstacle
			obstacle = Obstacle(x, y, obs_width, obs_height)
			
			# Check if it overlaps with existing obstacles
			if not self._obstacle_overlaps(obstacle):
				self.obstacles.append(obstacle)
				current_obstacle_area += obs_width * obs_height
			
			attempts += 1
	
	def _obstacle_overlaps(self, new_obstacle: Obstacle) -> bool:
		"""Check if a new obstacle overlaps with existing ones."""
		for obstacle in self.obstacles:
			if self._rectangles_overlap(new_obstacle, obstacle):
				return True
		return False
	
	def _rectangles_overlap(self, rect1: Obstacle, rect2: Obstacle) -> bool:
		"""Check if two rectangles overlap."""
		x1_min, y1_min, x1_max, y1_max = rect1.get_bounds()
		x2_min, y2_min, x2_max, y2_max = rect2.get_bounds()
		
		return not (x1_max < x2_min or x2_max < x1_min or
				   y1_max < y2_min or y2_max < y1_min)
	
	def set_start_goal(self, 
					   start: Tuple[float, float], 
					   goal: Tuple[float, float]):
		"""
		Set start and goal configurations.
		
		Args:
			start: Start configuration (x, y)
			goal: Goal configuration (x, y)
		"""
		if not self._is_valid_config(start):
			raise ValueError(f"Start configuration {start} is invalid")
		if not self._is_valid_config(goal):
			raise ValueError(f"Goal configuration {goal} is invalid")
		
		self.start_config = start
		self.goal_config = goal
	
	def _is_valid_config(self, config: Tuple[float, float]) -> bool:
		"""Check if a configuration is valid (within bounds and not in obstacle)."""
		x, y = config
		width, height = self.dimensions
		
		# Check bounds
		if not (0 <= x <= width and 0 <= y <= height):
			return False
		
		# Check collision with obstacles
		for obstacle in self.obstacles:
			if obstacle.contains(config):
				return False
		
		return True
	
	def is_path_collision_free(self, 
								path: List[Tuple[float, float]], 
								step_size: float = 0.1) -> bool:
		"""
		Check if a path is collision-free.
		
		Args:
			path: List of waypoints defining the path
			step_size: Step size for collision checking along the path
			
		Returns:
			True if path is collision-free, False otherwise
		"""
		if len(path) < 2:
			return True
		
		for i in range(len(path) - 1):
			start_point = path[i]
			end_point = path[i + 1]
			
			# Check line segment for collisions
			if not self._is_line_segment_collision_free(start_point, end_point, step_size):
				return False
		
		return True
	
	def _is_line_segment_collision_free(self, 
									   start: Tuple[float, float], 
									   end: Tuple[float, float], 
									   step_size: float) -> bool:
		"""Check if a line segment is collision-free."""
		start_array = np.array(start)
		end_array = np.array(end)
		direction = end_array - start_array
		distance = np.linalg.norm(direction)
		
		if distance == 0:
			return self._is_valid_config(start)
		
		direction_normalized = direction / distance
		num_steps = int(distance / step_size) + 1
		
		for i in range(num_steps + 1):
			t = i / num_steps
			point = start_array + t * direction
			if not self._is_valid_config(tuple(point)):
				return False
		
		return True
	
	def get_environment_stats(self) -> Dict[str, Any]:
		"""Get statistics about the environment."""
		width, height = self.dimensions
		total_area = width * height
		
		obstacle_areas = [obs.width * obs.height for obs in self.obstacles]
		total_obstacle_area = sum(obstacle_areas)
		actual_density = total_obstacle_area / total_area
		
		return {
			"dimensions": self.dimensions,
			"total_area": total_area,
			"num_obstacles": len(self.obstacles),
			"total_obstacle_area": total_obstacle_area,
			"target_density": self.obstacle_density,
			"actual_density": actual_density,
			"start_config": self.start_config,
			"goal_config": self.goal_config
		}
	
	def visualize(self, ax=None):
		"""Visualize the environment (placeholder for now)."""
		# This will be implemented in the visualization module
		pass 