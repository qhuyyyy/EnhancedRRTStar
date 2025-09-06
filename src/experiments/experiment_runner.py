"""
Experiment runner for comprehensive RRT algorithm comparisons.

This module provides functionality to run experiments comparing:
- Standard RRT
- Standard RRT*
- Informed RRT*
- Enhanced RRT* (our proposed algorithm)
"""

import time
import numpy as np
# pandas will be imported when needed
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Imports will be handled at runtime


class ExperimentRunner:
    """
    Comprehensive experiment runner for RRT algorithm comparisons.
    
    This class provides methods to:
    - Run multiple algorithms on various environments
    - Collect performance metrics
    - Generate comparison plots
    - Export results for analysis
    """
    
    def __init__(self, 
                 output_dir: str = "results",
                 random_seed: Optional[int] = 42):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory to save experiment results
            random_seed: Random seed for reproducible experiments
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Algorithm configurations will be imported dynamically
        self.algorithms = {}
        self._import_algorithms()
        
        # Environment configurations (as specified in the paper)
        self.environments = {
            'Easy': {'dimensions': (15, 10), 'obstacle_density': 0.123},    # 12.3% density
            'Medium': {'dimensions': (25, 20), 'obstacle_density': 0.287},   # 28.7% density
            'Hard': {'dimensions': (35, 30), 'obstacle_density': 0.452}    # 45.2% density
        }
        
        # Start/goal configurations for each environment
        self.start_goal_configs = {
            'Easy': ((1, 1), (13, 8)),
            'Medium': ((1, 1), (22, 18)),
            'Hard': ((1, 1), (33, 28))  # Adjusted for 35x30 environment
        }
        
        # Performance metrics to collect
        self.metrics = [
            'success_rate', 'computation_time', 'path_length', 
            'tree_size', 'path_accuracy', 'iterations'
        ]

    def _create_fixed_hard_obstacles(self):
        """Create a fixed hard map with guaranteed connectivity."""
        from src.environments.environment import Obstacle
        obstacles = []
        # Create obstacles for 35x30 environment with guaranteed path from (1,1) to (33,28)
        # Simple obstacles that don't block the main diagonal path
        obstacles += [
            # Small scattered obstacles
            Obstacle(5, 5, 3, 2),      # Obstacle 1
            Obstacle(10, 8, 2, 3),     # Obstacle 2
            Obstacle(15, 12, 3, 2),    # Obstacle 3
            Obstacle(20, 15, 2, 2),    # Obstacle 4
            Obstacle(25, 18, 3, 2),    # Obstacle 5
            Obstacle(8, 20, 2, 3),     # Obstacle 6
            Obstacle(18, 22, 3, 2),    # Obstacle 7
            Obstacle(28, 25, 2, 2),    # Obstacle 8
            # Some larger obstacles for complexity
            Obstacle(12, 3, 4, 2),     # Large obstacle 1
            Obstacle(22, 8, 3, 3),     # Large obstacle 2
            Obstacle(6, 15, 3, 2),     # Large obstacle 3
            Obstacle(26, 20, 4, 2),    # Large obstacle 4
        ]
        return obstacles
    
    def _ensure_valid_start_goal(self, env, start, goal):
        """Ensure start and goal configurations are valid (not in obstacles)."""
        # Check if start is valid
        if not env._is_valid_config(start):
            # Find nearest valid start
            start = self._find_nearest_valid_config(env, start)
        
        # Check if goal is valid
        if not env._is_valid_config(goal):
            # Find nearest valid goal
            goal = self._find_nearest_valid_config(env, goal)
        
        return start, goal
    
    def _find_nearest_valid_config(self, env, config):
        """Find the nearest valid configuration to the given config."""
        x, y = config
        width, height = env.dimensions
        
        # Try expanding in a spiral pattern
        for radius in range(1, min(width, height)):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        new_x = max(0, min(width, x + dx))
                        new_y = max(0, min(height, y + dy))
                        new_config = (new_x, new_y)
                        
                        if env._is_valid_config(new_config):
                            return new_config
        
        # Fallback: return a corner
        return (1, 1)
    
    def _import_algorithms(self):
        """Import RRT algorithms dynamically."""
        try:
            from ..algorithms import StandardRRT, StandardRRTStar, InformedRRTStar, EnhancedRRT
            self.algorithms = {
                'RRT': StandardRRT,
                'RRT*': StandardRRTStar,
                'Informed RRT*': InformedRRTStar,
                'Enhanced RRT*': EnhancedRRT
            }
        except ImportError:
            # Fallback to basic imports
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.algorithms import StandardRRT, StandardRRTStar, InformedRRTStar, EnhancedRRT
            self.algorithms = {
                'RRT': StandardRRT,
                'RRT*': StandardRRTStar,
                'Informed RRT*': InformedRRTStar,
                'Enhanced RRT*': EnhancedRRT
            }
    
    def run_single_experiment(self, 
                              algorithm_class,
                              environment_config: Dict,
                              start_goal: Tuple[Tuple[float, float], Tuple[float, float]],
                              max_iterations: int = 8000,  # Increased default iterations
                              num_trials: int = 10) -> Dict[str, Any]:
        """
        Run a single experiment configuration.
        
        Args:
            algorithm_class: RRT algorithm class to test
            environment_config: Environment configuration
            start_goal: Start and goal configurations
            max_iterations: Maximum iterations per trial
            num_trials: Number of trials to run
            
        Returns:
            Dictionary containing experiment results
        """
        # Import Environment dynamically
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.environments.environment import Environment
        
        start, goal = start_goal
        results = {
            'algorithm': algorithm_class.__name__,
            'environment': environment_config,
            'start': start,
            'goal': goal,
            'max_iterations': max_iterations,
            'num_trials': num_trials,
            'trials': []
        }
        
        # Use fixed obstacles for Hard environment to ensure connectivity
        use_fixed_hard = (environment_config['dimensions'] == (35, 30) and 
                         environment_config['obstacle_density'] == 0.452)
        fixed_hard_obstacles = self._create_fixed_hard_obstacles() if use_fixed_hard else None
        
        for trial in range(num_trials):
            # Create fresh environment
            env = Environment(
                dimensions=environment_config['dimensions'],
                obstacle_density=environment_config['obstacle_density'],
                random_seed=42,  # Use fixed seed for reproducibility
                obstacles=fixed_hard_obstacles
            )
            
            # Ensure start and goal are valid
            valid_start, valid_goal = self._ensure_valid_start_goal(env, start, goal)
            
            # Create planner
            planner = algorithm_class(
                environment=env,
                max_iterations=max_iterations,
                random_seed=np.random.randint(10000)
            )
            
            # Run planning
            start_time = time.time()
            path = planner.plan(valid_start, valid_goal)
            end_time = time.time()
            
            # Collect trial results
            trial_result = {
                'trial': trial,
                'success': len(path) > 0,
                'computation_time': end_time - start_time,
                'path_length': self._calculate_path_length(path),
                'tree_size': len(planner.nodes),
                'iterations': planner.iteration,
                'path_accuracy': self._calculate_path_accuracy(path, start, goal)
            }
            
            results['trials'].append(trial_result)
        
        # Calculate aggregate statistics
        results.update(self._calculate_aggregate_stats(results['trials']))
        
        return results
    
    def run_comprehensive_comparison(self, 
                                    max_iterations: int = 5000,
                                    num_trials: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive comparison of all algorithms on all environments.
        
        Args:
            max_iterations: Maximum iterations per trial
            num_trials: Number of trials per configuration
            
        Returns:
            Dictionary containing all experiment results
        """
        all_results = {
            'experiment_config': {
                'max_iterations': max_iterations,
                'num_trials': num_trials,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': []
        }
        
        # Run experiments for each algorithm and environment
        for env_name, env_config in self.environments.items():
            start_goal = self.start_goal_configs[env_name]
            
            for alg_name, alg_class in self.algorithms.items():
                print(f"Running {alg_name} on {env_name} environment...")
                
                result = self.run_single_experiment(
                    algorithm_class=alg_class,
                    environment_config=env_config,
                    start_goal=start_goal,
                    max_iterations=max_iterations,
                    num_trials=num_trials
                )
                
                all_results['results'].append(result)
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
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
    
    def _calculate_path_accuracy(self, 
                                 path: List[Tuple[float, float]], 
                                 start: Tuple[float, float], 
                                 goal: Tuple[float, float]) -> float:
        """Calculate path accuracy (optimal path length / actual path length)."""
        if len(path) == 0:
            return 0.0
        
        # Calculate actual path length
        actual_length = self._calculate_path_length(path)
        
        # Calculate optimal path length (Euclidean distance)
        optimal_length = np.linalg.norm(np.array(goal) - np.array(start))
        
        # Path accuracy = optimal / actual (higher is better, closer to 1.0)
        return optimal_length / actual_length if actual_length > 0 else 0.0
    
    def _calculate_aggregate_stats(self, trials: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics from trial results."""
        if not trials:
            return {}
        
        # Extract metrics
        success_rates = [t['success'] for t in trials]
        computation_times = [t['computation_time'] for t in trials]
        path_lengths = [t['path_length'] for t in trials]
        tree_sizes = [t['tree_size'] for t in trials]
        iterations = [t['iterations'] for t in trials]
        path_accuracies = [t['path_accuracy'] for t in trials]
        
        # Calculate statistics
        stats = {
            'success_rate': np.mean(success_rates),
            'computation_time_mean': np.mean(computation_times),
            'computation_time_std': np.std(computation_times),
            'path_length_mean': np.mean(path_lengths),
            'path_length_std': np.std(path_lengths),
            'tree_size_mean': np.mean(tree_sizes),
            'tree_size_std': np.std(tree_sizes),
            'iterations_mean': np.mean(iterations),
            'iterations_std': np.std(iterations),
            'path_accuracy_mean': np.mean(path_accuracies),
            'path_accuracy_std': np.std(path_accuracies)
        }
        
        return stats
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results to files."""
        # Save as JSON
        import json
        json_path = self.output_dir / "experiment_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save as CSV for analysis
        csv_data = []
        for result in results['results']:
            row = {
                'algorithm': result['algorithm'],
                'environment': f"{result['environment']['dimensions'][0]}x{result['environment']['dimensions'][1]}",
                'obstacle_density': result['environment']['obstacle_density'],
                **{k: v for k, v in result.items() if k not in ['algorithm', 'environment', 'start', 'goal', 'max_iterations', 'num_trials', 'trials']}
            }
            csv_data.append(row)
        
        try:
            import pandas as pd
            df = pd.DataFrame(csv_data)
        except ImportError:
            # If pandas is not available, save as simple CSV
            import csv
            csv_path = self.output_dir / "experiment_results.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                if csv_data:
                    writer.writerow(csv_data[0].keys())
                    for row in csv_data:
                        writer.writerow(row.values())
            print(f"Results saved to {csv_path}")
            return
        csv_path = self.output_dir / "experiment_results.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {json_path} and {csv_path}")
    
    def generate_comparison_plots(self, results: Dict[str, Any]):
        """Generate comparison plots from experiment results."""
        # Import Visualizer dynamically
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.visualization.visualizer import Visualizer
        visualizer = Visualizer()
        
        # Create performance comparison plots
        self._create_performance_plots(results, visualizer)
        
        # Create environment-specific plots
        self._create_environment_plots(results, visualizer)
    
    def _create_performance_plots(self, results: Dict[str, Any], visualizer):
        """Create performance comparison plots."""
        # Extract data for plotting
        algorithms = []
        success_rates = []
        computation_times = []
        path_lengths = []
        tree_sizes = []
        
        for result in results['results']:
            algorithms.append(result['algorithm'])
            success_rates.append(result['success_rate'])
            computation_times.append(result['computation_time_mean'])
            path_lengths.append(result['path_length_mean'])
            tree_sizes.append(result['tree_size_mean'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Success rate comparison
        axes[0].bar(algorithms, success_rates, color=['blue', 'green', 'orange', 'red'])
        axes[0].set_title('Success Rate Comparison')
        axes[0].set_ylabel('Success Rate')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Computation time comparison
        axes[1].bar(algorithms, computation_times, color=['blue', 'green', 'orange', 'red'])
        axes[1].set_title('Computation Time Comparison')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Path length comparison
        axes[2].bar(algorithms, path_lengths, color=['blue', 'green', 'orange', 'red'])
        axes[2].set_title('Path Length Comparison')
        axes[2].set_ylabel('Path Length')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Tree size comparison
        axes[3].bar(algorithms, tree_sizes, color=['blue', 'green', 'orange', 'red'])
        axes[3].set_title('Tree Size Comparison')
        axes[3].set_ylabel('Number of Nodes')
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved to {plot_path}")
    
    def _create_environment_plots(self, results: Dict[str, Any], visualizer):
        """Create environment-specific comparison plots."""
        # Group results by environment
        env_results = {}
        for result in results['results']:
            env_key = f"{result['environment']['dimensions'][0]}x{result['environment']['dimensions'][1]}"
            if env_key not in env_results:
                env_results[env_key] = []
            env_results[env_key].append(result)
        
        # Create plots for each environment
        for env_name, env_data in env_results.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            # Extract metrics
            algorithms = [r['algorithm'] for r in env_data]
            success_rates = [r['success_rate'] for r in env_data]
            computation_times = [r['computation_time_mean'] for r in env_data]
            path_lengths = [r['path_length_mean'] for r in env_data]
            tree_sizes = [r['tree_size_mean'] for r in env_data]
            
            # Create subplots
            axes[0].bar(algorithms, success_rates, color=['blue', 'green', 'orange', 'red'])
            axes[0].set_title(f'{env_name} - Success Rate')
            axes[0].set_ylabel('Success Rate')
            axes[0].tick_params(axis='x', rotation=45)
            
            axes[1].bar(algorithms, computation_times, color=['blue', 'green', 'orange', 'red'])
            axes[1].set_title(f'{env_name} - Computation Time')
            axes[1].set_ylabel('Time (seconds)')
            axes[1].tick_params(axis='x', rotation=45)
            
            axes[2].bar(algorithms, path_lengths, color=['blue', 'green', 'orange', 'red'])
            axes[2].set_title(f'{env_name} - Path Length')
            axes[2].set_ylabel('Path Length')
            axes[2].tick_params(axis='x', rotation=45)
            
            axes[3].bar(algorithms, tree_sizes, color=['blue', 'green', 'orange', 'red'])
            axes[3].set_title(f'{env_name} - Tree Size')
            axes[3].set_ylabel('Number of Nodes')
            axes[3].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"{env_name}_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Environment comparison plot saved to {plot_path}")


# Import matplotlib for plotting
import matplotlib.pyplot as plt 