"""
Hyperparameter grid search script.

Performs exhaustive grid search over specified hyperparameter ranges.
Runs validation episodes for each combination and saves results.

Usage:
    # Classical grid search
    python scripts/hyperparameter_search.py --mode classical \\
        --lr_values 0.001 0.005 0.01 --gamma_values 0.95 0.99 \\
        --episodes 100 --seeds 42 123

    # Quantum grid search
    python scripts/hyperparameter_search.py --mode quantum \\
        --lr_values 0.005 0.01 0.02 --depth_values 2 3 4 \\
        --episodes 100 --seeds 42
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import itertools
from tqdm import tqdm

from src.models import TinyMLP, QuantumPolicy
from src.agent import REINFORCEAgent
from src.env_wrapper import create_env
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter grid search')
    
    parser.add_argument('--mode', type=str, required=True, choices=['classical', 'quantum'],
                        help='Training mode')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Episodes per configuration')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds to test (space-separated)')
    
    # Hyperparameter ranges
    parser.add_argument('--lr_values', type=float, nargs='+',
                        default=[0.005, 0.01, 0.02],
                        help='Learning rates to test')
    parser.add_argument('--gamma_values', type=float, nargs='+',
                        default=[0.95, 0.99],
                        help='Gamma values to test')
    parser.add_argument('--grad_clip_values', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0],
                        help='Gradient clipping values to test')
    
    # Quantum-specific
    parser.add_argument('--depth_values', type=int, nargs='+',
                        default=[2, 3],
                        help='Circuit depths to test (quantum only)')
    parser.add_argument('--diff_method', type=str, default='backprop',
                        choices=['backprop', 'parameter-shift'],
                        help='Gradient method for quantum (default: backprop for speed)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/hyperparameter_search',
                        help='Output directory for results')
    
    return parser.parse_args()


def set_seeds(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_single_config(mode, config_dict, episodes, seed):
    """
    Run training with a single hyperparameter configuration.
    
    Args:
        mode: 'classical' or 'quantum'
        config_dict: Dictionary of hyperparameters
        episodes: Number of training episodes
        seed: Random seed
        
    Returns:
        dict: Results including rewards and statistics
    """
    set_seeds(seed)
    
    # Create model
    if mode == 'classical':
        model = TinyMLP()
    else:
        model = QuantumPolicy(
            n_qubits=4,
            n_layers=config_dict.get('depth', 3),
            measurement='softmax',
            diff_method=config_dict.get('diff_method', 'backprop')
        )
    
    # Create agent
    agent = REINFORCEAgent(
        model,
        lr=config_dict['lr'],
        gamma=config_dict['gamma'],
        grad_clip=config_dict['grad_clip'],
        use_scheduler=False  # Disable for consistency
    )
    
    # Create environment
    env = create_env()
    
    # Train
    episode_rewards = agent.train(
        env,
        episodes=episodes,
        log_interval=episodes + 1,  # No logging during search
        seed=seed
    )
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards, ddof=1)
    final_20_avg = np.mean(episode_rewards[-20:])
    max_reward = np.max(episode_rewards)
    
    # Check if improving
    first_half = np.mean(episode_rewards[:len(episode_rewards)//2])
    second_half = np.mean(episode_rewards[len(episode_rewards)//2:])
    improvement = second_half - first_half
    
    return {
        'config': config_dict,
        'seed': seed,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'final_20_avg': final_20_avg,
        'max_reward': max_reward,
        'improvement': improvement,
        'episode_rewards': episode_rewards
    }


def grid_search(args):
    """Perform grid search over hyperparameters."""
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER GRID SEARCH - {args.mode.upper()}")
    print(f"{'='*70}")
    print(f"Episodes per config: {args.episodes}")
    print(f"Seeds: {args.seeds}")
    print(f"Learning rates: {args.lr_values}")
    print(f"Gamma values: {args.gamma_values}")
    print(f"Grad clip values: {args.grad_clip_values}")
    
    if args.mode == 'quantum':
        print(f"Circuit depths: {args.depth_values}")
    
    # Generate all combinations
    if args.mode == 'classical':
        param_grid = {
            'lr': args.lr_values,
            'gamma': args.gamma_values,
            'grad_clip': args.grad_clip_values
        }
    else:
        param_grid = {
            'lr': args.lr_values,
            'gamma': args.gamma_values,
            'grad_clip': args.grad_clip_values,
            'depth': args.depth_values,
            'diff_method': [args.diff_method]  # Use specified diff_method for all configs
        }
    
    # Create all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*values))
    
    configs = [dict(zip(keys, combo)) for combo in combinations]
    
    total_runs = len(configs) * len(args.seeds)
    print(f"\nTotal configurations: {len(configs)}")
    print(f"Total seeds: {len(args.seeds)}")
    print(f"Total runs: {total_runs}")
    print(f"{'='*70}\n")
    
    # Run all configurations
    all_results = []
    
    pbar = tqdm(total=total_runs, desc="Grid search", unit="run")
    
    for config in configs:
        for seed in args.seeds:
            result = run_single_config(args.mode, config, args.episodes, seed)
            all_results.append(result)
            
            # Update progress bar with best so far
            best_so_far = max(all_results, key=lambda x: x['mean_reward'])
            pbar.set_postfix({
                'best_mean': f"{best_so_far['mean_reward']:.1f}",
                'best_lr': f"{best_so_far['config']['lr']}"
            })
            pbar.update(1)
    
    pbar.close()
    
    # Aggregate results by configuration (average across seeds)
    config_results = {}
    for result in all_results:
        config_key = tuple(sorted(result['config'].items()))
        if config_key not in config_results:
            config_results[config_key] = []
        config_results[config_key].append(result)
    
    # Compute aggregated statistics
    aggregated = []
    for config_key, results in config_results.items():
        config_dict = dict(config_key)
        mean_rewards = [r['mean_reward'] for r in results]
        final_20_avgs = [r['final_20_avg'] for r in results]
        
        aggregated.append({
            'config': config_dict,
            'mean_reward_avg': np.mean(mean_rewards),
            'mean_reward_std': np.std(mean_rewards, ddof=1) if len(mean_rewards) > 1 else 0,
            'final_20_avg': np.mean(final_20_avgs),
            'final_20_std': np.std(final_20_avgs, ddof=1) if len(final_20_avgs) > 1 else 0,
            'n_seeds': len(results)
        })
    
    # Sort by mean reward
    aggregated.sort(key=lambda x: x['mean_reward_avg'], reverse=True)
    
    # Print top 5
    print(f"\n{'='*70}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*70}")
    
    for i, result in enumerate(aggregated[:5], 1):
        print(f"\n{i}. Mean Reward: {result['mean_reward_avg']:.2f} ± {result['mean_reward_std']:.2f}")
        print(f"   Final 20-ep avg: {result['final_20_avg']:.2f} ± {result['final_20_std']:.2f}")
        print(f"   Configuration:")
        for key, value in result['config'].items():
            print(f"      {key}: {value}")
    
    print(f"\n{'='*70}\n")
    
    # Save results
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all results
    results_file = output_dir / 'grid_search_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'all_results': [{k: v for k, v in r.items() if k != 'episode_rewards'} 
                           for r in all_results],
            'aggregated': aggregated,
            'top_config': aggregated[0]['config']
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Save best configuration separately
    best_config_file = output_dir / 'best_config.json'
    with open(best_config_file, 'w') as f:
        json.dump(aggregated[0]['config'], f, indent=2)
    
    print(f"Best configuration saved to {best_config_file}")
    
    return aggregated[0]


def main():
    args = parse_args()
    
    # Validate
    if args.episodes < 50:
        print("⚠️  Warning: Very short validation runs may not be representative")
    
    if len(args.seeds) > 5:
        print("⚠️  Warning: Many seeds will significantly increase runtime")
    
    best = grid_search(args)
    
    print("\n🎯 RECOMMENDED CONFIGURATION:")
    for key, value in best['config'].items():
        print(f"   --{key} {value}")
    print()


if __name__ == '__main__':
    main()
