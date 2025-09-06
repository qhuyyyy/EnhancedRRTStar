#!/usr/bin/env python3
"""
Create comparison plots for all three environment types (Easy, Medium, Hard)
with the same style as demo_comparison.png
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environments.environment import Environment
from algorithms import StandardRRT, StandardRRTStar, InformedRRTStar, EnhancedRRT
from visualization.visualizer import Visualizer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def create_three_maps_comparison():
    """Create comparison plots for all three environment types."""
    
    # Environment configurations
    env_configs = {
        'Easy': {
            'dimensions': (15, 10),
            'obstacle_density': 0.123,
            'start': (1, 1),
            'goal': (13, 8),
            'title': 'Easy Environment (15×10)'
        },
        'Medium': {
            'dimensions': (25, 20),
            'obstacle_density': 0.20,
            'start': (1, 1),
            'goal': (22, 18),
            'title': 'Medium Environment (25×20)'
        },
        'Hard': {
            'dimensions': (35, 30),
            'obstacle_density': 0.287,
            'start': (1, 1),
            'goal': (34, 28),
            'title': 'Hard Environment (35×30)'
        }
    }
    
    # Create the main figure with 3 rows (one for each environment)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for env_idx, (env_name, config) in enumerate(env_configs.items()):
        print(f"Processing {env_name} environment...")
        
        # Create environment
        env = Environment(
            dimensions=config['dimensions'],
            obstacle_density=config['obstacle_density'],
            random_seed=42
        )
        
        env.set_start_goal(config['start'], config['goal'])
        
        # Create planners
        planners = {
            'RRT': StandardRRT(env, max_iterations=3000, random_seed=42),
            'RRT*': StandardRRTStar(env, max_iterations=3000, random_seed=42),
            'Informed RRT*': InformedRRTStar(env, max_iterations=3000, random_seed=42),
            'Enhanced RRT*': (
                EnhancedRRT(env, max_iterations=3000, random_seed=42)
                if env_name != 'Hard'
                else EnhancedRRT(
                    env,
                    max_iterations=3000,
                    step_size=1.55,         # keep near current for stability
                    goal_bias=0.10,         # lower bias → more exploration → more nodes
                    rewiring_radius=2.3,    # slightly smaller radius
                    adaptive_sampling_factor=1.6,
                    sampling_bias_coefficient=2.5,
                    random_seed=42,
                )
            )
        }
        
        # Run planning for each algorithm
        results = {}
        for alg_name, planner in planners.items():
            print(f"  Running {alg_name}...")
            path = planner.plan(config['start'], config['goal'])
            
            results[alg_name] = {
                'success': len(path) > 0,
                'path_length': len(path),
                'tree_size': len(planner.nodes),
                'iterations': planner.iteration,
                'path': path
            }
            
            print(f"    Success: {results[alg_name]['success']}")
            print(f"    Path Length: {results[alg_name]['path_length']}")
            print(f"    Tree Size: {results[alg_name]['tree_size']}")
            print(f"    Iterations: {results[alg_name]['iterations']}")
        
        # Plot each algorithm in the row
        for alg_idx, (alg_name, result) in enumerate(results.items()):
            ax = axes[env_idx, alg_idx]
            
            # Plot environment background
            for obstacle in env.obstacles:
                rect = patches.Rectangle(
                    (obstacle.x, obstacle.y), 
                    obstacle.width, 
                    obstacle.height,
                    linewidth=1, 
                    edgecolor='black',
                    facecolor='black',
                    alpha=0.7
                )
                ax.add_patch(rect)
            
            # Plot tree edges (darker blue)
            if result['success']:
                # Reduce visual prominence for Enhanced RRT* on Hard to reduce clutter
                is_hard_enhanced = (env_name == 'Hard' and alg_name == 'Enhanced RRT*')
                if is_hard_enhanced:
                    # Plot all tree edges with original blue color and alpha, using default line width like other algorithms
                    for node in planners[alg_name].nodes:
                        if node.parent is not None:
                            x_coords = [node.parent.config[0], node.config[0]]
                            y_coords = [node.parent.config[1], node.config[1]]
                            ax.plot(x_coords, y_coords, 
                                   color='blue', 
                                   alpha=0.6,
                                   linewidth=1.0)  # default line width like other algorithms
                else:
                    # Plot all edges for other algorithms
                    for node in planners[alg_name].nodes:
                        if node.parent is not None:
                            x_coords = [node.parent.config[0], node.config[0]]
                            y_coords = [node.parent.config[1], node.config[1]]
                            ax.plot(x_coords, y_coords, 
                                   color='blue', 
                                   alpha=0.6, 
                                   linewidth=1.0)
            
            # Plot path (thick orange with dots)
            if result['success'] and result['path']:
                path = result['path']
                x_coords = [point[0] for point in path]
                y_coords = [point[1] for point in path]
                is_hard_enhanced = (env_name == 'Hard' and alg_name == 'Enhanced RRT*')
                if is_hard_enhanced:
                    ax.plot(x_coords, y_coords, 
                           color='orange', 
                           linewidth=3)
                else:
                    ax.plot(x_coords, y_coords, 
                           color='orange', 
                           linewidth=3, 
                           marker='o', 
                           markersize=4)
            
            # Plot start and goal
            ax.plot(config['start'][0], config['start'][1], 'o', 
                   color='green', markersize=8, 
                   markeredgecolor='black', markeredgewidth=2)
            ax.plot(config['goal'][0], config['goal'][1], 'o', 
                   color='red', markersize=8, 
                   markeredgecolor='black', markeredgewidth=2)
            
            # Set subplot properties with proper axis limits
            ax.set_xlim(0, config['dimensions'][0])
            ax.set_ylim(0, config['dimensions'][1])
            
            # Set proper tick marks for axes
            x_ticks = np.arange(0, config['dimensions'][0] + 1, 5)
            y_ticks = np.arange(0, config['dimensions'][1] + 1, 5)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            
            # Create title with statistics
            if result['success']:
                title = f"{alg_name}"
            else:
                title = f"{alg_name}\nNo path found"
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Add environment title for the row (slightly closer than default)
        fig.text(0.5, 0.945 - env_idx * 0.315, config['title'], 
                fontsize=14, ha='center', weight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "three_maps_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nThree maps comparison plot saved to {output_path}")
    
    return fig


if __name__ == "__main__":
    try:
        fig = create_three_maps_comparison()
        print("\nThree maps comparison completed successfully!")
    except Exception as e:
        print(f"Error during three maps comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 