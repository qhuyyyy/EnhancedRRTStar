#!/usr/bin/env python3
"""
Simple demo script for Enhanced RRT* algorithm.

This script demonstrates the basic functionality of the Enhanced RRT* algorithm
on a simple environment.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environments.environment import Environment
from algorithms import EnhancedRRT
from visualization.visualizer import Visualizer


def main():
    """Run a simple demo of Enhanced RRT*."""
    print("Enhanced RRT* Demo")
    print("=" * 30)
    
    # Create a simple environment
    print("Creating environment...")
    env = Environment(dimensions=(20, 15), obstacle_density=0.2, random_seed=42)
    
    # Set start and goal
    start = (1, 1)
    goal = (18, 13)
    env.set_start_goal(start, goal)
    
    print(f"Environment: {env.dimensions[0]}x{env.dimensions[1]}")
    print(f"Obstacles: {len(env.obstacles)}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    
    # Create Enhanced RRT* planner
    print("\nCreating Enhanced RRT* planner...")
    planner = EnhancedRRT(
        environment=env,
        max_iterations=2000,
        step_size=2.0,  # Updated to match paper
        goal_bias=0.1,
        goal_tolerance=1.0,  # Updated to match paper
        random_seed=42
    )
    
    # Plan path
    print("Planning path...")
    path = planner.plan(start, goal)
    
    # Print results
    print(f"\nResults:")
    print(f"  Path found: {len(path) > 0}")
    print(f"  Path length: {len(path)} waypoints")
    print(f"  Tree size: {len(planner.nodes)} nodes")
    print(f"  Iterations: {planner.iteration}")
    
    if path:
        print(f"  Path: {path[:3]}...{path[-3:] if len(path) > 6 else path}")
    
    # Get enhanced statistics
    stats = planner.get_enhanced_stats()
    print(f"\nEnhanced RRT* Statistics:")
    print(f"  Adaptive goal bias: {stats['adaptive_goal_bias']:.3f}")
    print(f"  Adaptive sampling factor: {stats['adaptive_sampling_factor']}")
    print(f"  Sampling bias coefficient: {stats['sampling_bias_coefficient']}")
    
    # Create visualization
    print("\nCreating visualization...")
    visualizer = Visualizer()
    
    # Plot the environment
    fig_env = visualizer.plot_environment(
        env, 
        title="Enhanced RRT* Demo Environment",
        save_path="demo_environment.png"
    )
    
    # Plot the RRT tree
    fig_tree = visualizer.plot_rrt_tree(
        planner,
        title="Enhanced RRT* Tree and Path",
        save_path="demo_tree.png"
    )
    
    print("Visualizations saved:")
    print("  - demo_environment.png")
    print("  - demo_tree.png")
    
    return path


if __name__ == "__main__":
    try:
        path = main()
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 