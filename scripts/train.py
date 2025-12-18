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

from src.models import TinyMLP, QuantumPolicy
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
    
    # Setup periodic saving callback
    def save_checkpoint_callback(episode, stats):
        """Save intermediate results every 50 episodes."""
        import json
        import os
        
        # Save rewards incrementally
        results_dir = Path('results') / args.mode
        results_dir.mkdir(parents=True, exist_ok=True)
        rewards_path = results_dir / f'seed{args.seed}_rewards.json'
        
        # Atomic write to prevent corruption
        temp_path = str(rewards_path) + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump({
                "seed": args.seed,
                "mode": args.mode,
                "episode_rewards": stats.episode_rewards,
                "num_episodes": episode
            }, f, indent=2)
        os.replace(temp_path, rewards_path)
        print(f"  [Checkpoint saved: episode {episode}]")
    
    # Train with periodic checkpointing
    stats = agent.train(num_episodes=args.episodes, env=env, verbose=True, 
                        save_callback=save_checkpoint_callback, seed=args.seed)
    
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
    """Train quantum VQC agent."""
    print(f"\n{'='*60}")
    print(f"TRAINING QUANTUM AGENT")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Circuit depth: {args.depth} layers")
    print(f"Gradient method: {args.diff_method}")
    print(f"{'='*60}\n")
    
    # Create quantum policy
    # Note: diff_method is handled by PennyLane's default.qubit device
    # 'backprop' is automatic for simulators, 'parameter-shift' requires explicit QNode config
    policy = QuantumPolicy(n_qubits=4, n_layers=args.depth, measurement='softmax')
    print(f"Model parameters: {policy.count_parameters()}")
    
    # Create agent
    agent = REINFORCEAgent(policy, lr=args.lr, gamma=args.gamma)
    
    # Create environment
    env = create_env()
    
    # Setup periodic saving callback
    def save_checkpoint_callback(episode, stats):
        """Save intermediate results every 50 episodes."""
        import json
        import os
        
        # Save rewards incrementally
        results_dir = Path('results') / args.mode
        results_dir.mkdir(parents=True, exist_ok=True)
        rewards_path = results_dir / f'seed{args.seed}_rewards.json'
        
        # Atomic write to prevent corruption
        temp_path = str(rewards_path) + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump({
                "seed": args.seed,
                "mode": args.mode,
                "episode_rewards": stats.episode_rewards,
                "num_episodes": episode
            }, f, indent=2)
        os.replace(temp_path, rewards_path)
        print(f"  [Checkpoint saved: episode {episode}]")
    
    # Train
    print(f"Starting training with {args.diff_method} gradients...")
    if args.diff_method == 'parameter-shift':
        print("WARNING: Parameter-shift is slow. Consider using backprop for prototyping.")
    
    stats = agent.train(num_episodes=args.episodes, env=env, verbose=True,
                        save_callback=save_checkpoint_callback, seed=args.seed)
    
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
