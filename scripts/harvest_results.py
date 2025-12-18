#!/usr/bin/env python3
"""
Harvest partial results from ongoing hyperparameter search and training runs.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime


def harvest_hyperparam_search(log_file='logs/hyperparam_search.log'):
    """Extract completed configurations from hyperparameter search log."""
    print("="*80)
    print("HYPERPARAMETER SEARCH HARVEST")
    print("="*80)
    
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return []
    
    with open(log_file) as f:
        lines = f.readlines()
    
    # Parse completed runs
    configs = []
    current_config = {}
    
    for i, line in enumerate(lines):
        # Look for config start
        if 'Config' in line and 'lr=' in line:
            # Extract parameters using regex
            lr_match = re.search(r'lr=([\d.]+)', line)
            depth_match = re.search(r'depth=(\d+)', line)
            gamma_match = re.search(r'gamma=([\d.]+)', line)
            clip_match = re.search(r'grad_clip=([\d.]+)', line)
            
            if all([lr_match, depth_match, gamma_match, clip_match]):
                current_config = {
                    'lr': float(lr_match.group(1)),
                    'depth': int(depth_match.group(1)),
                    'gamma': float(gamma_match.group(1)),
                    'grad_clip': float(clip_match.group(1))
                }
        
        # Look for results
        if 'Mean:' in line or 'mean_reward' in line:
            mean_match = re.search(r'(Mean|mean_reward)[:\s]+([\d.]+)', line)
            if mean_match and current_config:
                current_config['mean_reward'] = float(mean_match.group(2))
                configs.append(current_config.copy())
                current_config = {}
    
    if not configs:
        print("No completed configurations found yet")
        print(f"Log file size: {Path(log_file).stat().st_size / 1024:.1f} KB")
        return []
    
    # Sort by performance
    configs.sort(key=lambda x: x.get('mean_reward', 0), reverse=True)
    
    print(f"\nFound {len(configs)} completed configurations\n")
    print("TOP 5 CONFIGURATIONS:")
    print("-"*80)
    
    for i, config in enumerate(configs[:5], 1):
        print(f"{i}. Mean Reward: {config['mean_reward']:.1f}")
        print(f"   LR={config['lr']}, Depth={config['depth']}, "
              f"Gamma={config['gamma']}, GradClip={config['grad_clip']}")
        print()
    
    # Save best to JSON
    if configs:
        best_config_file = 'results/best_hyperparams_partial.json'
        with open(best_config_file, 'w') as f:
            json.dump(configs[0], f, indent=2)
        print(f"✅ Best config saved to: {best_config_file}")
    
    return configs


def check_training_progress(seed, log_file=None):
    """Check progress of ongoing training run."""
    if log_file is None:
        log_file = f'logs/param_shift_seed{seed}_true.log'
    
    print(f"\n{'='*80}")
    print(f"SEED {seed} PARAMETER-SHIFT PROGRESS")
    print("="*80)
    
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return
    
    # Get latest progress from log
    with open(log_file) as f:
        lines = f.readlines()
    
    # Find latest training line
    training_lines = [l for l in lines if 'Training:' in l]
    if training_lines:
        latest = training_lines[-1]
        print(f"Latest status: {latest.strip()}")
        
        # Parse progress
        episode_match = re.search(r'(\d+)/(\d+)', latest)
        reward_match = re.search(r'reward=([\d.]+)', latest)
        avg_match = re.search(r'avg_\d+=([\d.]+)', latest)
        
        if episode_match:
            current, total = episode_match.groups()
            progress = int(current) / int(total) * 100
            print(f"\nProgress: {current}/{total} ({progress:.1f}%)")
        
        if reward_match:
            print(f"Current episode reward: {reward_match.group(1)}")
        
        if avg_match:
            print(f"Recent average: {avg_match.group(1)}")
    
    # Check results file
    results_file = f'results/quantum/seed{seed}_rewards.json'
    if Path(results_file).exists():
        with open(results_file) as f:
            data = json.load(f)
        
        rewards = data.get('episode_rewards', [])
        print(f"\nResults file: {len(rewards)} episodes saved")
        
        if len(rewards) >= 100:
            final_100 = sum(rewards[-100:]) / 100
            print(f"Last 100-ep average: {final_100:.1f}")


def main():
    """Main harvest function."""
    print(f"\nHarvesting results at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Harvest hyperparameter search
    configs = harvest_hyperparam_search()
    
    # Check ongoing parameter-shift runs
    for seed in [2, 17]:
        check_training_progress(seed)
    
    print("\n" + "="*80)
    print("HARVEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
