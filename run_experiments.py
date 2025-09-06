#!/usr/bin/env python3
"""
Main script to run comprehensive RRT algorithm comparison experiments.

This script demonstrates the Enhanced RRT* algorithm and compares it with:
- Standard RRT
- Standard RRT*
- Informed RRT*

Usage:
    python run_experiments.py [--trials N] [--iterations N] [--output-dir DIR]
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiments.experiment_runner import ExperimentRunner
from visualization.visualizer import Visualizer
from environments.environment import Environment
from algorithms import StandardRRT, StandardRRTStar, InformedRRTStar, EnhancedRRT


def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive RRT algorithm comparison experiments"
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=10,
        help="Number of trials per configuration (default: 10)"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=5000,
        help="Maximum iterations per trial (default: 5000)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--quick-demo", 
        action="store_true",
        help="Run a quick demonstration with fewer iterations"
    )
    
    args = parser.parse_args()
    
    print("Enhanced RRT* Algorithm Comparison")
    print("=" * 50)
    print(f"Trials per configuration: {args.trials}")
    print(f"Max iterations per trial: {args.iterations}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Adjust parameters for quick demo
    if args.quick_demo:
        args.iterations = 1000
        args.trials = 5
        print("Quick demo mode: reduced iterations and trials")
        print()
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir=args.output_dir, random_seed=42)
    
    try:
        # Run comprehensive comparison
        print("Starting comprehensive algorithm comparison...")
        results = runner.run_comprehensive_comparison(
            max_iterations=args.iterations,
            num_trials=args.trials
        )
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to {args.output_dir}/")
        
        # Generate comparison plots
        print("\nGenerating comparison plots...")
        runner.generate_comparison_plots(results)
        
        # Print summary statistics
        print("\n" + "=" * 50)
        print("EXPERIMENT SUMMARY")
        print("=" * 50)
        
        for result in results['results']:
            alg_name = result['algorithm']
            env_name = f"{result['environment']['dimensions'][0]}x{result['environment']['dimensions'][1]}"
            success_rate = result['success_rate']
            comp_time = result['computation_time_mean']
            path_length = result['path_length_mean']
            tree_size = result['tree_size_mean']
            
            print(f"\n{alg_name} on {env_name} environment:")
            print(f"  Success Rate: {success_rate:.3f}")
            print(f"  Computation Time: {comp_time:.3f}s")
            print(f"  Path Length: {path_length:.3f}")
            print(f"  Tree Size: {tree_size:.0f} nodes")
        
        print("\n" + "=" * 50)
        print("All results and plots have been saved!")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_single_demo():
    """Run a single demonstration of Enhanced RRT*."""
    print("Enhanced RRT* Single Demo")
    print("=" * 30)
    
    # Create environment
    env = Environment(dimensions=(25, 20), obstacle_density=0.287, random_seed=42)
    start, goal = (1, 1), (22, 18)
    env.set_start_goal(start, goal)
    
    # Create planners
    planners = {
        'RRT': StandardRRT(env, max_iterations=2000, random_seed=42),
        'RRT*': StandardRRTStar(env, max_iterations=2000, random_seed=42),
        'Informed RRT*': InformedRRTStar(env, max_iterations=2000, random_seed=42),
        'Enhanced RRT*': EnhancedRRT(env, max_iterations=2000, random_seed=42)
    }
    
    # Run planning
    results = {}
    for name, planner in planners.items():
        print(f"\nRunning {name}...")
        path = planner.plan(start, goal)
        
        results[name] = {
            'success': len(path) > 0,
            'path_length': len(path),
            'tree_size': len(planner.nodes),
            'iterations': planner.iteration
        }
        
        print(f"  Success: {results[name]['success']}")
        print(f"  Path Length: {results[name]['path_length']}")
        print(f"  Tree Size: {results[name]['tree_size']}")
        print(f"  Iterations: {results[name]['iterations']}")
    
    # Create visualization
    visualizer = Visualizer()
    
    # Create comparison plot
    fig = visualizer.plot_algorithm_comparison(
        planners, env, start, goal,
        title="Enhanced RRT* Algorithm Comparison Demo"
    )
    
    # Save plot
    output_path = "demo_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDemo comparison plot saved to {output_path}")
    
    return results


if __name__ == "__main__":
    # Check if quick demo is requested
    if "--quick-demo" in sys.argv:
        print("Running quick demo...")
        run_single_demo()
    else:
        # Run full experiments
        exit_code = main()
        sys.exit(exit_code) 