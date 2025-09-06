#!/usr/bin/env python3
"""
Create three additional figures for the research paper:
1. Path Corridor Heatmap - showing path stability across multiple runs
2. Tree Growth Snapshots - showing how trees evolve during planning
3. Clearance-Colored Path - showing path safety distance to obstacles
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environments.environment import Environment
from algorithms import StandardRRT, StandardRRTStar, InformedRRTStar, EnhancedRRT
from visualization.visualizer import Visualizer
import numpy as np


def create_paper_figures():
    """Create three additional figures for the research paper."""
    
    print("Creating three additional figures for the research paper...")
    
    # Create visualizer
    visualizer = Visualizer()
    
    # Environment configuration (using Medium for demonstration)
    env = Environment(
        dimensions=(25, 20),
        obstacle_density=0.20,
        random_seed=42
    )
    
    start = (1, 1)
    goal = (22, 18)
    env.set_start_goal(start, goal)
    
    # Create planners
    planners = {
        'RRT': StandardRRT(env, max_iterations=2000, random_seed=42),
        'RRT*': StandardRRTStar(env, max_iterations=2000, random_seed=42),
        'Informed RRT*': InformedRRTStar(env, max_iterations=2000, random_seed=42),
        'Enhanced RRT*': EnhancedRRT(env, max_iterations=2000, random_seed=42)
    }
    
    print("\n1. Creating Path Corridor Heatmap...")
    print("   This will run 50 iterations for each algorithm...")
    
    # Figure 1: Path Corridor Heatmap
    fig1 = visualizer.create_path_corridor_heatmap(
        environment=env,
        planners=planners,
        start=start,
        goal=goal,
        num_runs=50,
        title="Path Corridor Analysis - Medium Environment (25×20)"
    )
    fig1.savefig("paper_figure_1_path_corridor.png", dpi=300, bbox_inches='tight')
    print("   ✓ Saved: paper_figure_1_path_corridor.png")
    
    print("\n2. Creating Tree Growth Snapshots...")
    
    # Create separate planners with higher max_iterations for Figure 2
    snapshot_planners = {
        'RRT': StandardRRT(env, max_iterations=8000, random_seed=42),
        'RRT*': StandardRRTStar(env, max_iterations=8000, random_seed=42),
        'Informed RRT*': InformedRRTStar(env, max_iterations=8000, random_seed=42),
        'Enhanced RRT*': EnhancedRRT(env, max_iterations=8000, random_seed=42)
    }
    
    # Figure 2: Tree Growth Snapshots
    fig2 = visualizer.create_tree_growth_snapshots(
        environment=env,
        planners=snapshot_planners,
        start=start,
        goal=goal,
        snapshot_percentages=[25, 50, 75, 100],
        title="Tree Growth Evolution - Medium Environment (25×20)"
    )
    fig2.savefig("paper_figure_2_tree_growth.png", dpi=300, bbox_inches='tight')
    print("   ✓ Saved: paper_figure_2_tree_growth.png")
    
    print("\n3. Creating Clearance-Colored Path...")
    
    # Figure 3: Clearance-Colored Path
    fig3 = visualizer.create_clearance_colored_path(
        environment=env,
        planners=planners,
        start=start,
        goal=goal,
        title="Path Safety Analysis - Medium Environment (25×20)"
    )
    fig3.savefig("paper_figure_3_clearance_safety.png", dpi=300, bbox_inches='tight')
    print("   ✓ Saved: paper_figure_3_clearance_safety.png")
    
    print("\n4. Creating Path Quality Distribution...")
    print("   This will run 100 iterations for statistical analysis...")
    
    # Figure 4: Path Quality Distribution
    fig4 = visualizer.create_path_quality_distribution(
        environment=env,
        planners=planners,
        start=start,
        goal=goal,
        num_runs=100,
        title="Path Quality Distribution - Medium Environment (25×20)"
    )
    fig4.savefig("paper_figure_4_quality_distribution.png", dpi=300, bbox_inches='tight')
    print("   ✓ Saved: paper_figure_4_quality_distribution.png")
    
    print("\n" + "="*60)
    print("All four paper figures created successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("1. paper_figure_1_path_corridor.png - Path stability analysis")
    print("2. paper_figure_2_tree_growth.png - Tree evolution snapshots")
    print("3. paper_figure_3_clearance_safety.png - Path safety analysis")
    print("4. paper_figure_4_quality_distribution.png - Path quality distribution")
    print("\nThese figures demonstrate:")
    print("• Enhanced RRT* path consistency across multiple runs")
    print("• Intelligent tree exploration behavior")
    print("• Improved path safety and obstacle avoidance")
    print("• Superior path quality reliability and stability")
    
    return [fig1, fig2, fig3, fig4]


if __name__ == "__main__":
    try:
        figures = create_paper_figures()
        print("\nPaper figures generation completed successfully!")
    except Exception as e:
        print(f"Error during paper figures generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 