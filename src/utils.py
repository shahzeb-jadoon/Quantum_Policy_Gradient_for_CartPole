"""
Utility functions for logging, plotting, and result management.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_training_results(rewards, seed, mode, output_dir='results'):
    """
    Save training rewards to JSON file.
    
    Args:
        rewards (list): Episode rewards
        seed (int): Random seed used
        mode (str): Training mode ('classical' or 'quantum')
        output_dir (str): Output directory
    """
    # Create output directory
    output_path = Path(output_dir) / mode
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save rewards as JSON
    results = {
        'seed': seed,
        'mode': mode,
        'episode_rewards': rewards,
        'num_episodes': len(rewards)
    }
    
    filename = output_path / f'seed{seed}_rewards.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")


def plot_training_curve(rewards, seed, mode, output_dir='results', window=100):
    """
    Plot training curve with moving average.
    
    Args:
        rewards (list): Episode rewards
        seed (int): Random seed
        mode (str): Training mode ('classical' or 'quantum')
        output_dir (str): Output directory
        window (int): Window size for moving average
    """
    # Create output directory
    output_path = Path(output_dir) / mode
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate moving average
    moving_avg = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(rewards[start_idx:i+1]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot raw rewards (semi-transparent)
    ax.plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    
    # Plot moving average
    ax.plot(moving_avg, color='darkblue', linewidth=2, label=f'{window}-Episode Average')
    
    # Add solved threshold line
    ax.axhline(y=195, color='red', linestyle='--', linewidth=1.5, label='Solved Threshold (195)')
    
    # Labels and title
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'{mode.capitalize()} Agent Training - Seed {seed}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    # Save plot
    filename = output_path / f'seed{seed}_plot.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {filename}")


def print_summary(rewards, mode):
    """
    Print training summary statistics.
    
    Args:
        rewards (list): Episode rewards
        mode (str): Training mode
    """
    print("\n" + "="*60)
    print(f"{mode.upper()} AGENT TRAINING SUMMARY")
    print("="*60)
    
    # Final 100-episode average
    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    print(f"Final 100-episode average: {final_avg:.1f}")
    
    # Best episode
    best_reward = max(rewards)
    best_episode = rewards.index(best_reward) + 1
    print(f"Best episode: {best_episode} (reward: {best_reward:.0f})")
    
    # Check if solved
    if len(rewards) >= 100 and final_avg >= 195:
        # Find when it first solved
        for i in range(99, len(rewards)):
            avg = np.mean(rewards[i-99:i+1])
            if avg >= 195:
                print(f"✓ Solved at episode {i+1}!")
                break
    else:
        print("✗ Not solved (avg reward < 195 over 100 episodes)")
    
    print("="*60 + "\n")
