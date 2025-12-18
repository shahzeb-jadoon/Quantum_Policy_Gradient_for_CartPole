"""
Quick validation script for testing hyperparameters.

Runs shorter training episodes (50-100) for rapid iteration.
Useful for hyperparameter tuning before full training runs.

Usage:
    python scripts/validate.py --mode classical --lr 0.01 --episodes 100
    python scripts/validate.py --mode quantum --depth 2 --episodes 100
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.models import TinyMLP, QuantumPolicy
from src.agent import REINFORCEAgent
from src.env_wrapper import create_env
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quick validation for hyperparameter testing')
    
    parser.add_argument('--mode', type=str, required=True, choices=['classical', 'quantum'],
                        help='Training mode: classical or quantum')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of validation episodes (50-200 recommended)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate to test')
    parser.add_argument('--gamma', type=float, default=config.GAMMA,
                        help=f'Discount factor (default: {config.GAMMA})')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    
    # Quantum-specific
    parser.add_argument('--depth', type=int, default=config.QUANTUM_LAYERS,
                        help=f'Quantum circuit depth (default: {config.QUANTUM_LAYERS})')
    
    # Disable scheduler by default for validation
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Enable adaptive learning rate scheduler')
    
    return parser.parse_args()


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def validate(args):
    """Run validation with specified hyperparameters."""
    print(f"\n{'='*60}")
    print(f"VALIDATION RUN - {args.mode.upper()}")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Grad clip: {args.grad_clip}")
    print(f"Use scheduler: {args.use_scheduler}")
    
    # Set seeds
    set_seeds(args.seed)
    
    # Create model
    if args.mode == 'classical':
        model = TinyMLP()
    else:
        print(f"Circuit depth: {args.depth} layers")
        model = QuantumPolicy(n_qubits=4, n_layers=args.depth, measurement='softmax')
    
    print(f"Model parameters: {model.count_parameters()}")
    print(f"{'='*60}\n")
    
    # Create agent
    agent = REINFORCEAgent(
        model,
        lr=args.lr,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        use_scheduler=args.use_scheduler
    )
    
    # Create environment
    env = create_env()
    
    # Train
    episode_rewards = agent.train(
        env,
        episodes=args.episodes,
        log_interval=max(10, args.episodes // 5),  # Log 5 times during run
        seed=args.seed
    )
    
    # Quick stats
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards, ddof=1):.2f}")
    print(f"Last 20 avg: {np.mean(episode_rewards[-20:]):.2f}")
    print(f"Best episode: {max(episode_rewards):.0f}")
    print(f"Worst episode: {min(episode_rewards):.0f}")
    
    # Check if trending upward
    first_half = np.mean(episode_rewards[:len(episode_rewards)//2])
    second_half = np.mean(episode_rewards[len(episode_rewards)//2:])
    improvement = second_half - first_half
    print(f"Improvement (2nd half - 1st half): {improvement:+.2f}")
    
    if improvement > 0:
        print("✓ Trending upward - good signs!")
    else:
        print("✗ Trending downward - may need hyperparameter adjustment")
    
    print(f"{'='*60}\n")
    
    return episode_rewards


def main():
    args = parse_args()
    
    # Validate episode count
    if args.episodes > 200:
        print("⚠️  Warning: validation runs should be quick (50-200 episodes)")
        print("   For full training, use scripts/train.py instead\n")
    
    validate(args)


if __name__ == '__main__':
    main()
