# Enhanced RRT\*: An Improved Rapidly-Exploring Random Tree Algorithm

## Overview

This repository contains the implementation of Enhanced RRT*, an advanced variant of the RRT* algorithm designed to overcome the practical and theoretical limitations of classical sampling-based motion planners. The algorithm introduces three key innovations:

1. **Adaptive Obstacle-Aware Sampling**: Dynamically adjusts sampling density based on local obstacle distribution
2. **Intelligent Dynamic Rewiring**: Optimizes tree connectivity using adaptive radius adjustment
3. **Goal-Directed Exploration with Adaptive Bias**: Balances exploration and exploitation for faster convergence

## Project Structure

```
RRT/
├── src/                    # Source code
│   ├── algorithms/        # RRT algorithm implementations
│   ├── environments/      # Environment generation and management
│   ├── experiments/       # Experiment runner and comparison
│   ├── visualization/     # Plotting and visualization tools
│   └── utils/            # Utility functions and helpers
├── tests/                 # Unit tests and validation
├── data/                  # Test data and configurations
├── results/               # Experimental results and outputs
├── figures/               # Generated figures and plots
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
├── run_experiments.py     # Main experiment script
├── demo.py               # Simple demo script
└── README.md             # This file
```

### Performance Metrics

The experiments measure the following metrics:

- **Success Rate**: Percentage of successful path planning attempts
- **Computation Time**: Time required to find a solution (seconds)
- **Path Length**: Number of waypoints in the planned path
- **Tree Size**: Number of nodes in the final tree
- **Path Accuracy**: Ratio of optimal path length to actual path length

## Algorithms Implemented

This repository includes implementations of four RRT algorithms:

1. **Standard RRT**: Basic rapidly-exploring random tree algorithm
2. **Standard RRT\***: Asymptotically optimal RRT with rewiring
3. **Informed RRT\***: RRT\* with ellipsoidal sampling region
4. **Enhanced RRT\***: Our proposed algorithm with three key innovations
