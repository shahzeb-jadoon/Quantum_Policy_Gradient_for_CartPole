"""
Training script for CartPole policy gradient agents.

Usage:
    python scripts/train.py --mode classical --seed 42 --episodes 500
    python scripts/train.py --mode quantum --seed 1 --episodes 500 --diff_method backprop
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.models import TinyMLP
from src.agent import REINFORCEAgent
from src.env_wrapper import create_env
from src.utils import save_training_results, plot_training_curve, print_summary
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CartPole policy gradient agent')
    
    parser.add_argument('--mode', type=str, required=True, choices=['classical', 'quantum'],
                        help='Training mode: classical (TinyMLP) or quantum (VQC)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default from config)')
    parser.add_argument('--gamma', type=float, default=config.GAMMA,
                        help=f'Discount factor (default: {config.GAMMA})')
    
    # Quantum-specific arguments (for Phase 3)
    parser.add_argument('--diff_method', type=str, default='backprop',
                        choices=['backprop', 'parameter-shift'],
                        help='Gradient method for quantum (backprop or parameter-shift)')
    parser.add_argument('--depth', type=int, default=config.QUANTUM_LAYERS,
                        help=f'Quantum circuit depth (default: {config.QUANTUM_LAYERS})')
    parser.add_argument('--shots', type=int, default=config.QUANTUM_SHOTS,
                        help=f'Quantum shots per measurement (default: {config.QUANTUM_SHOTS})')
    
    return parser.parse_args()


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_classical(args):
    """Train classical TinyMLP agent."""
    print(f"\n{'='*60}")
    print(f"TRAINING CLASSICAL AGENT")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"{'='*60}\n")
    
    # Create policy and agent
    policy = TinyMLP()
    print(f"Model parameters: {policy.count_parameters()}")
    
    agent = REINFORCEAgent(policy, lr=args.lr, gamma=args.gamma)
    
    # Create environment
    env = create_env()
    
    # Train
    stats = agent.train(num_episodes=args.episodes, env=env, verbose=True)
    
    # Save results
    save_training_results(stats.episode_rewards, args.seed, args.mode)
    plot_training_curve(stats.episode_rewards, args.seed, args.mode)
    print_summary(stats.episode_rewards, args.mode)
    
    # Save model checkpoint
    checkpoint_dir = Path('checkpoints') / args.mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'seed{args.seed}_final.pth'
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


def train_quantum(args):
    """Train quantum VQC agent (placeholder for Phase 3)."""
    print("\n" + "="*60)
    print("QUANTUM TRAINING NOT YET IMPLEMENTED")
    print("="*60)
    print("Quantum agent will be implemented in Phase 2-3.")
    print("For now, use --mode classical to train the baseline.")
    print("="*60 + "\n")
    raise NotImplementedError("Quantum training coming in Phase 3")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set default learning rates from config if not specified
    if args.lr is None:
        if args.mode == 'classical':
            args.lr = config.LR_CLASSICAL
        else:
            args.lr = config.LR_QUANTUM
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Train based on mode
    if args.mode == 'classical':
        train_classical(args)
    elif args.mode == 'quantum':
        train_quantum(args)


if __name__ == '__main__':
    main()
