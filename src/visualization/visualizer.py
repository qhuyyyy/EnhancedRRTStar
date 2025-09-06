"""
Main visualization class for RRT algorithms.

This module provides comprehensive visualization capabilities for:
- Environment visualization with obstacles
- RRT tree visualization
- Path visualization
- Multi-algorithm comparison plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
# Import will be handled at runtime


class Visualizer:
    """
    Main visualization class for RRT algorithms.
    
    This class provides methods to visualize:
    - Environments with obstacles
    - RRT trees and their evolution
    - Final paths and their quality
    - Algorithm performance comparisons
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'start': 'green',
            'goal': 'red',
            'tree': 'blue',
            'path': 'orange',
            'obstacles': 'black',
            'nodes': 'darkblue'
        }
    
    def plot_environment(self, 
                        environment,
                        show_start_goal: bool = True,
                        title: str = "Environment",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the environment with obstacles.
        
        Args:
            environment: Environment object to visualize
            show_start_goal: Whether to show start and goal configurations
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot obstacles
        for obstacle in environment.obstacles:
            rect = patches.Rectangle(
                (obstacle.x, obstacle.y), 
                obstacle.width, 
                obstacle.height,
                linewidth=1, 
                edgecolor=self.colors['obstacles'],
                facecolor=self.colors['obstacles'],
                alpha=0.7
            )
            ax.add_patch(rect)
        
        # Plot start and goal if available
        if show_start_goal and environment.start_config:
            ax.plot(environment.start_config[0], environment.start_config[1], 
                   'o', color=self.colors['start'], markersize=10, 
                   label='Start', markeredgecolor='black', markeredgewidth=2)
        
        if show_start_goal and environment.goal_config:
            ax.plot(environment.goal_config[0], environment.goal_config[1], 
                   'o', color=self.colors['goal'], markersize=10, 
                   label='Goal', markeredgecolor='black', markeredgewidth=2)
        
        # Set plot properties
        ax.set_xlim(0, environment.dimensions[0])
        ax.set_ylim(0, environment.dimensions[1])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_rrt_tree(self, 
                      planner,
                      show_path: bool = True,
                      show_nodes: bool = True,
                      title: str = "RRT Tree",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the RRT tree with optional path highlighting.
        
        Args:
            planner: RRT planner object
            show_path: Whether to highlight the final path
            show_nodes: Whether to show individual nodes
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot environment first
        self._plot_environment_background(ax, planner.environment)
        
        # Plot tree edges
        self._plot_tree_edges(ax, planner.nodes)
        
        # Plot nodes if requested
        if show_nodes:
            self._plot_tree_nodes(ax, planner.nodes)
        
        # Plot path if requested and available
        if show_path and planner.final_path:
            self._plot_path(ax, planner.final_path)
        
        # Set plot properties
        ax.set_xlim(0, planner.environment.dimensions[0])
        ax.set_ylim(0, planner.environment.dimensions[1])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_algorithm_comparison(self, 
                                 planners: Dict[str, Any],
                                 environment,
                                 start: Tuple[float, float],
                                 goal: Tuple[float, float],
                                 title: str = "Algorithm Comparison",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple RRT algorithms.
        
        Args:
            planners: Dictionary of planner objects
            environment: Environment object
            start: Start configuration
            goal: Goal configuration
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot each algorithm
        for i, (name, planner) in enumerate(planners.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Plot environment background
            self._plot_environment_background(ax, environment)
            
            # Plan path if not already planned
            if not planner.path_found:
                planner.plan(start, goal)
            
            # Plot tree and path
            self._plot_tree_edges(ax, planner.nodes, alpha=0.3)
            if planner.final_path:
                self._plot_path(ax, planner.final_path, linewidth=3)
            
            # Plot start and goal
            ax.plot(start[0], start[1], 'o', color=self.colors['start'], 
                   markersize=8, markeredgecolor='black', markeredgewidth=2)
            ax.plot(goal[0], goal[1], 'o', color=self.colors['goal'], 
                   markersize=8, markeredgecolor='black', markeredgewidth=2)
            
            # Set subplot properties
            ax.set_xlim(0, environment.dimensions[0])
            ax.set_ylim(0, environment.dimensions[1])
            ax.set_title(f"{name}\nNodes: {len(planner.nodes)}, "
                        f"Path Length: {self._calculate_path_length(planner.final_path):.2f}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for i in range(len(planners), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_environment_background(self, ax, environment):
        """Plot environment obstacles as background."""
        for obstacle in environment.obstacles:
            rect = patches.Rectangle(
                (obstacle.x, obstacle.y), 
                obstacle.width, 
                obstacle.height,
                linewidth=1, 
                edgecolor=self.colors['obstacles'],
                facecolor=self.colors['obstacles'],
                alpha=0.7
            )
            ax.add_patch(rect)
    
    def _plot_tree_edges(self, ax, nodes, alpha: float = 0.6):
        """Plot tree edges connecting parent and child nodes."""
        for node in nodes:
            if node.parent is not None:
                x_coords = [node.parent.config[0], node.config[0]]
                y_coords = [node.parent.config[1], node.config[1]]
                ax.plot(x_coords, y_coords, 
                       color=self.colors['tree'], 
                       alpha=alpha, 
                       linewidth=0.5)
    
    def _plot_tree_nodes(self, ax, nodes):
        """Plot individual tree nodes."""
        x_coords = [node.config[0] for node in nodes]
        y_coords = [node.config[1] for node in nodes]
        ax.plot(x_coords, y_coords, 'o', 
               color=self.colors['nodes'], 
               markersize=2, 
               alpha=0.7)
    
    def _plot_path(self, ax, path: List[Tuple[float, float]], linewidth: int = 2):
        """Plot the final path."""
        if len(path) < 2:
            return
        
        x_coords = [point[0] for point in path]
        y_coords = [point[1] for point in path]
        ax.plot(x_coords, y_coords, 
               color=self.colors['path'], 
               linewidth=linewidth, 
               marker='o', 
               markersize=4)
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate the total length of a path."""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += np.linalg.norm(
                np.array(path[i + 1]) - np.array(path[i])
            )
        
        return total_length
    
    def create_enhanced_single_layout(self, 
                                    environment_configs: Dict[str, Tuple],
                                    save_path: str = "enhanced_single_layout.png") -> plt.Figure:
        """
        Create the enhanced single layout figure as shown in the paper.
        
        Args:
            environment_configs: Dictionary of environment configurations
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Environment configurations
        env_types = ['easy', 'medium', 'hard']
        titles = ['Easy Environment (15×10, 12.3% density)', 
                 'Medium Environment (25×20, 28.7% density)',
                 'Hard Environment (35×30, 45.2% density)']
        
        for i, (env_type, title) in enumerate(zip(env_types, titles)):
            ax = axes[i]
            
            # Create environment
            from ..environments.environment import Environment
            if env_type == 'easy':
                env = Environment(dimensions=(15, 10), obstacle_density=0.123)
            elif env_type == 'medium':
                env = Environment(dimensions=(25, 20), obstacle_density=0.287)
            else:  # hard
                env = Environment(dimensions=(35, 30), obstacle_density=0.452)
            
            # Set start and goal
            if env_type == 'easy':
                start, goal = (1, 2), (13, 8)
            elif env_type == 'medium':
                start, goal = (2, 2.5), (22, 18)
            else:  # hard
                start, goal = (2, 2.5), (32, 27)
            
            env.set_start_goal(start, goal)
            
            # Plot environment
            self._plot_environment_background(ax, env)
            
            # Plot start and goal
            ax.plot(start[0], start[1], 'o', color=self.colors['start'], 
                   markersize=8, markeredgecolor='black', markeredgewidth=2)
            ax.plot(goal[0], goal[1], 'o', color=self.colors['goal'], 
                   markersize=8, markeredgecolor='black', markeredgewidth=2)
            
            # Set subplot properties
            ax.set_xlim(0, env.dimensions[0])
            ax.set_ylim(0, env.dimensions[1])
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Hide the fourth subplot
        axes[3].set_visible(False)
        
        plt.suptitle("Enhanced RRT* Environment Configurations", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 

    def create_path_corridor_heatmap(self, 
                                    environment,
                                    planners: Dict[str, Any],
                                    start: Tuple[float, float],
                                    goal: Tuple[float, float],
                                    num_runs: int = 50,
                                    random_seeds: Optional[List[int]] = None,
                                    title: str = "Path Corridor Analysis",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create path corridor heatmap by aggregating multiple path runs.
        
        Args:
            environment: Environment object
            planners: Dictionary of planner objects
            start: Start configuration
            goal: Goal configuration
            num_runs: Number of runs to aggregate
            random_seeds: List of random seeds (if None, will generate)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if random_seeds is None:
            random_seeds = list(range(42, 42 + num_runs))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Collect paths for each algorithm
        all_paths = {}
        for name in planners.keys():
            all_paths[name] = []
        
        print(f"Running {num_runs} iterations for each algorithm...")
        for i, seed in enumerate(random_seeds):
            if i % 10 == 0:
                print(f"  Progress: {i}/{num_runs}")
            
            # Set random seed for environment and planners
            np.random.seed(seed)
            
            # Run each planner
            for name, planner in planners.items():
                try:
                    # Reset planner
                    planner.reset()
                    # Plan path
                    path = planner.plan(start, goal)
                    if path and len(path) > 0:
                        all_paths[name].append(path)
                except Exception as e:
                    print(f"Warning: {name} failed on seed {seed}: {e}")
                    continue
        
        # Create heatmaps
        for idx, (name, paths) in enumerate(all_paths.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Plot environment
            self._plot_environment_background(ax, environment)
            
            # Plot start and goal
            ax.plot(start[0], start[1], 'o', color=self.colors['start'], 
                   markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax.plot(goal[0], goal[1], 'o', color=self.colors['goal'], 
                   markersize=10, markeredgecolor='black', markeredgewidth=2)
            
            if paths:
                # Create heatmap from all paths
                all_points = []
                for path in paths:
                    all_points.extend(path)
                
                if all_points:
                    # Convert to numpy array
                    points = np.array(all_points)
                    
                    # Create 2D histogram
                    width, height = environment.dimensions
                    bins_x = int(width * 2)  # 0.5 unit bins
                    bins_y = int(height * 2)
                    
                    hist, x_edges, y_edges = np.histogram2d(
                        points[:, 0], points[:, 1], 
                        bins=[bins_x, bins_y],
                        range=[[0, width], [0, height]]
                    )
                    
                    # Plot heatmap
                    extent = [0, width, 0, height]
                    im = ax.imshow(hist.T, origin='lower', extent=extent, 
                                 cmap='hot', alpha=0.7, aspect='auto')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Path Frequency', fontsize=10)
                
                # Simplified title: algorithm name only (remove Success and Avg Length)
                title_text = f"{name}"
                ax.set_title(title_text, fontsize=12, fontweight='bold')
            else:
                ax.set_title(f"{name}\nNo successful paths", fontsize=12, fontweight='bold')
            
            # Set plot properties
            ax.set_xlim(0, environment.dimensions[0])
            ax.set_ylim(0, environment.dimensions[1])
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for i in range(len(planners), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 

    def create_tree_growth_snapshots(self, 
                                   environment,
                                   planners: Dict[str, Any],
                                   start: Tuple[float, float],
                                   goal: Tuple[float, float],
                                   snapshot_percentages: List[float] = [25, 50, 75, 100],
                                   title: str = "Tree Growth Evolution",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create tree growth snapshots showing evolution at different iteration milestones.
        
        Args:
            environment: Environment object
            planners: Dictionary of planner objects
            start: Start configuration
            goal: Goal configuration
            snapshot_percentages: List of percentages to take snapshots at
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        num_planners = len(planners)
        num_snapshots = len(snapshot_percentages)
        
        fig, axes = plt.subplots(num_planners, num_snapshots, 
                                figsize=(4*num_snapshots, 4*num_planners))
        
        # Handle single planner case
        if num_planners == 1:
            axes = axes.reshape(1, -1)
        
        # Handle single snapshot case
        if num_snapshots == 1:
            axes = axes.reshape(-1, 1)
        
        # Run planning and capture snapshots
        for planner_idx, (name, planner) in enumerate(planners.items()):
            print(f"Capturing snapshots for {name}...")
            
            # Reset planner and initialize
            planner.reset()
            planner._initialize_planning(start, goal)
            
            # Calculate snapshot iterations
            max_iter = planner.max_iterations
            snapshot_iterations = [int(max_iter * p / 100) for p in snapshot_percentages]
            print(f"  Snapshot iterations: {snapshot_iterations}")
            
            tree_snapshots = []
            current_threshold_idx = 0
            iteration = 0
            
            # Force planning to continue through all thresholds even after finding path
            while iteration < max_iter and current_threshold_idx < len(snapshot_iterations):
                # Advance until we reach the next threshold
                while iteration < snapshot_iterations[current_threshold_idx]:
                    try:
                        planner._planning_iteration()
                        iteration += 1
                    except Exception as e:
                        print(f"Warning: {name} failed at iteration {iteration}: {e}")
                        break
                
                # Capture snapshot at this threshold
                snapshot = {
                    'nodes': planner.nodes.copy(),
                    'iteration': iteration,
                    'path_found': planner.path_found
                }
                tree_snapshots.append(snapshot)
                print(f"  Captured snapshot at iteration {iteration} with {len(planner.nodes)} nodes")
                current_threshold_idx += 1
            
            # Continue planning to completion for final snapshot
            while iteration < max_iter:
                try:
                    planner._planning_iteration()
                    iteration += 1
                except Exception as e:
                    print(f"Warning: {name} failed at iteration {iteration}: {e}")
                    break
            
            # Ensure we have exactly num_snapshots snapshots
            if len(tree_snapshots) > num_snapshots:
                tree_snapshots = tree_snapshots[:num_snapshots]
            while len(tree_snapshots) < num_snapshots:
                tree_snapshots.append(tree_snapshots[-1] if tree_snapshots else {'nodes': [], 'iteration': 0, 'path_found': False})
            
            # Plot snapshots
            for snapshot_idx in range(num_snapshots):
                snapshot = tree_snapshots[snapshot_idx]
                ax = axes[planner_idx, snapshot_idx]
                
                # Plot environment
                self._plot_environment_background(ax, environment)
                
                # Plot tree edges at this snapshot
                if snapshot['nodes']:
                    for node in snapshot['nodes']:
                        if node.parent is not None:
                            x_coords = [node.parent.config[0], node.config[0]]
                            y_coords = [node.parent.config[1], node.config[1]]
                            ax.plot(x_coords, y_coords, 
                                   color=self.colors['tree'], 
                                   alpha=0.6, 
                                   linewidth=0.8)
                
                # Plot start and goal
                ax.plot(start[0], start[1], 'o', color=self.colors['start'], 
                       markersize=8, markeredgecolor='black', markeredgewidth=2)
                ax.plot(goal[0], goal[1], 'o', color=self.colors['goal'], 
                       markersize=8, markeredgecolor='black', markeredgewidth=2)
                
                # Set title and properties
                percentage = snapshot_percentages[snapshot_idx]
                ax.set_title(f"{name}\n{percentage}%", fontsize=10, fontweight='bold')
                ax.set_xlim(0, environment.dimensions[0])
                ax.set_ylim(0, environment.dimensions[1])
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 

    def create_clearance_colored_path(self, 
                                    environment,
                                    planners: Dict[str, Any],
                                    start: Tuple[float, float],
                                    goal: Tuple[float, float],
                                    title: str = "Path Safety Analysis",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create clearance-colored paths showing safety distance to obstacles.
        
        Args:
            environment: Environment object
            planners: Dictionary of planner objects
            start: Start configuration
            goal: Goal configuration
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        num_planners = len(planners)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Define clearance color mapping
        def clearance_to_color(clearance):
            """Convert clearance distance to color (red=unsafe, green=safe)."""
            if clearance < 0.5:
                return 'red'  # Very unsafe
            elif clearance < 1.0:
                return 'orange'  # Unsafe
            elif clearance < 1.5:
                return 'yellow'  # Moderate
            elif clearance < 2.0:
                return 'lightgreen'  # Safe
            else:
                return 'green'  # Very safe
        
        def calculate_clearance(point, obstacles):
            """Calculate minimum distance from point to any obstacle."""
            min_distance = float('inf')
            for obstacle in obstacles:
                # Calculate distance to obstacle boundary
                dx = max(0, max(obstacle.x - point[0], point[0] - (obstacle.x + obstacle.width)))
                dy = max(0, max(obstacle.y - point[1], point[1] - (obstacle.y + obstacle.height)))
                
                if dx == 0 and dy == 0:  # Point inside obstacle
                    return 0.0
                
                distance = np.sqrt(dx*dx + dy*dy)
                min_distance = min(min_distance, distance)
            
            return min_distance
        
        # Plot each algorithm
        for idx, (name, planner) in enumerate(planners.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Plot environment
            self._plot_environment_background(ax, environment)
            
            # Plot start and goal
            ax.plot(start[0], start[1], 'o', color=self.colors['start'], 
                   markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax.plot(goal[0], goal[1], 'o', color=self.colors['goal'], 
                   markersize=10, markeredgecolor='black', markeredgewidth=2)
            
            # Plan path
            try:
                path = planner.plan(start, goal)
                
                if path and len(path) > 1:
                    # Plot path with clearance-based coloring
                    for i in range(len(path) - 1):
                        p1, p2 = path[i], path[i + 1]
                        
                        # Calculate clearance for both points
                        clearance1 = calculate_clearance(p1, environment.obstacles)
                        clearance2 = calculate_clearance(p2, environment.obstacles)
                        
                        # Use average clearance for the segment
                        avg_clearance = (clearance1 + clearance2) / 2
                        segment_color = clearance_to_color(avg_clearance)
                        
                        # Plot segment with clearance color
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                               color=segment_color, linewidth=4, alpha=0.8)
                    
                    # Add statistics
                    clearances = [calculate_clearance(p, environment.obstacles) for p in path]
                    avg_clearance = np.mean(clearances)
                    min_clearance = min(clearances)
                    path_length = len(path)
                    
                    title_text = f"{name}\nPath Length: {path_length}\nAvg Clearance: {avg_clearance:.2f}\nMin Clearance: {min_clearance:.2f}"
                    ax.set_title(title_text, fontsize=12, fontweight='bold')
                    
                    # Add legend for clearance colors
                    legend_elements = [
                        plt.Line2D([0], [0], color='red', linewidth=4, label='< 0.5 (Unsafe)'),
                        plt.Line2D([0], [0], color='orange', linewidth=4, label='0.5-1.0 (Unsafe)'),
                        plt.Line2D([0], [0], color='yellow', linewidth=4, label='1.0-1.5 (Moderate)'),
                        plt.Line2D([0], [0], color='lightgreen', linewidth=4, label='1.5-2.0 (Safe)'),
                        plt.Line2D([0], [0], color='green', linewidth=4, label='> 2.0 (Very Safe)')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                else:
                    ax.set_title(f"{name}\nNo path found", fontsize=12, fontweight='bold')
                    
            except Exception as e:
                ax.set_title(f"{name}\nError: {str(e)[:30]}...", fontsize=12, fontweight='bold')
            
            # Set plot properties
            ax.set_xlim(0, environment.dimensions[0])
            ax.set_ylim(0, environment.dimensions[1])
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for i in range(len(planners), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 

    def create_path_quality_distribution(self, 
                                       environment,
                                       planners: Dict[str, Any],
                                       start: Tuple[float, float],
                                       goal: Tuple[float, float],
                                       num_runs: int = 100,
                                       random_seeds: Optional[List[int]] = None,
                                       title: str = "Path Quality Distribution Analysis",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create path quality distribution analysis showing path length histograms and statistics.
        
        Args:
            environment: Environment object
            planners: Dictionary of planner objects
            start: Start configuration
            goal: Goal configuration
            num_runs: Number of runs to analyze
            random_seeds: List of random seeds (if None, will generate)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if random_seeds is None:
            random_seeds = list(range(42, 42 + num_runs))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Collect path lengths for each algorithm
        all_path_lengths = {}
        all_success_rates = {}
        
        for name in planners.keys():
            all_path_lengths[name] = []
            all_success_rates[name] = []
        
        print(f"Running {num_runs} iterations for path quality analysis...")
        for i, seed in enumerate(random_seeds):
            if i % 20 == 0:
                print(f"  Progress: {i}/{num_runs}")
            
            # Set random seed for environment and planners
            np.random.seed(seed)
            
            # Run each planner
            for name, planner in planners.items():
                try:
                    # Reset planner
                    planner.reset()
                    # Plan path
                    path = planner.plan(start, goal)
                    if path and len(path) > 0:
                        all_path_lengths[name].append(len(path))
                        all_success_rates[name].append(True)
                    else:
                        all_success_rates[name].append(False)
                except Exception as e:
                    print(f"Warning: {name} failed on seed {seed}: {e}")
                    all_success_rates[name].append(False)
                    continue
        
        # Calculate success rates
        for name in planners.keys():
            success_count = sum(1 for success in all_success_rates[name] if success)
            success_rate = success_count / num_runs * 100
            all_success_rates[name] = success_rate
        
        # Create distribution plots
        for idx, (name, path_lengths) in enumerate(all_path_lengths.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            if path_lengths:
                # Professional monochrome color scheme
                primary_color = '#2C3E50'      # Dark blue-gray (main color)
                accent_color = '#3498DB'       # Blue accent
                light_color = '#ECF0F1'        # Light gray
                text_color = '#2C3E50'         # Dark text
                
                # Create histogram with sophisticated styling
                n, bins, patches = ax.hist(path_lengths, bins=25, alpha=0.85, 
                                         color=primary_color,
                                         edgecolor='white', linewidth=1.5)
                
                # Add subtle gradient effect to histogram bars
                for i, patch in enumerate(patches):
                    patch.set_alpha(0.7 + (i % 3) * 0.1)  # Varying alpha for depth
                
                # Add statistics with improved formatting
                mean_length = np.mean(path_lengths)
                std_length = np.std(path_lengths)
                median_length = np.median(path_lengths)
                min_length = min(path_lengths)
                max_length = max(path_lengths)
                
                # Add vertical lines for mean and median with sophisticated styling
                ax.axvline(mean_length, color=accent_color, linestyle='-', linewidth=3, 
                          label=f'Mean: {mean_length:.1f}', alpha=0.9)
                ax.axvline(median_length, color=primary_color, linestyle='--', linewidth=2.5, 
                          label=f'Median: {median_length:.1f}', alpha=0.8)
                
                # Add enhanced text box with professional styling
                stats_text = f'Success Rate: {all_success_rates[name]:.1f}%\n'
                stats_text += f'Mean: {mean_length:.1f}\n'
                stats_text += f'Std: {std_length:.1f}\n'
                stats_text += f'Min: {min_length}\n'
                stats_text += f'Max: {max_length}\n'
                stats_text += f'Runs: {len(path_lengths)}'
                
                # Position text box with sophisticated styling
                ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                               alpha=0.98, edgecolor=primary_color, linewidth=2),
                       fontsize=11, fontweight='bold', color=text_color)
                
                # Set labels and title with professional styling
                ax.set_xlabel('Path Length (number of waypoints)', fontsize=14, fontweight='bold', 
                           color=text_color, labelpad=15)
                ax.set_ylabel('Frequency', fontsize=14, fontweight='bold', 
                           color=text_color, labelpad=15)
                ax.set_title(f'{name}\nPath Length Distribution', fontsize=16, fontweight='bold', 
                           color=text_color, pad=25)
                
                # Improve legend styling with professional appearance
                ax.legend(fontsize=12, framealpha=0.95, fancybox=True, shadow=True, 
                         loc='upper left', bbox_to_anchor=(0.02, 0.98),
                         facecolor='white', edgecolor=primary_color, frameon=True)
                
                # Improve grid and background with sophisticated design
                ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color=primary_color)
                ax.set_facecolor(light_color)
                
                # Improve axis styling for professional look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color(primary_color)
                ax.spines['bottom'].set_color(primary_color)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                
                # Set x-axis limits for better comparison with professional styling
                all_lengths = [length for lengths in all_path_lengths.values() for length in lengths if lengths]
                if all_lengths:
                    global_min = min(all_lengths)
                    global_max = max(all_lengths)
                    ax.set_xlim(global_min * 0.9, global_max * 1.1)
                
                # Improve tick styling for clarity
                ax.tick_params(axis='both', colors=text_color, labelsize=11, width=1.5, length=6)
                ax.tick_params(axis='x', pad=8)
                ax.tick_params(axis='y', pad=8)
                
                # Add subtle background pattern for sophistication
                ax.set_alpha(0.98)
                
            else:
                ax.set_title(f'{name}\nNo successful paths', fontsize=16, fontweight='bold', 
                           color=primary_color, pad=25)
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=20, color=text_color, fontweight='bold')
                ax.set_xlabel('Path Length', fontsize=14, fontweight='bold', color=text_color, labelpad=15)
                ax.set_ylabel('Frequency', fontsize=14, fontweight='bold', color=text_color, labelpad=15)
                ax.set_facecolor(light_color)
        
        # Hide unused subplots
        for i in range(len(planners), len(axes)):
            axes[i].set_visible(False)
        
        # Improve overall figure styling with professional appearance
        plt.suptitle(title, fontsize=20, fontweight='bold', color=text_color, y=0.98)
        plt.tight_layout()
        
        # Set figure background and improve overall appearance
        fig.patch.set_facecolor('white')
        fig.set_size_inches(18, 14)  # Slightly larger for better readability
        
        return fig 