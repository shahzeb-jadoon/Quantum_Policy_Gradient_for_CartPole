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
    
    # Training continuation
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--start_episode', type=int, default=0,
                        help='Episode number to resume from (for manual continuation)')
    
    return parser.parse_args()


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_checkpoint(agent, episode, rewards, optimizer, save_path):
    """
    Save complete training checkpoint.
    
    Args:
        agent: REINFORCE agent
        episode: Current episode number
        rewards: List of episode rewards so far
        optimizer: Optimizer state
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'model_state': agent.model.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'episode': episode,
        'rewards': rewards,
        'hyperparameters': {
            'lr': agent.lr,
            'gamma': agent.gamma,
            'grad_clip': agent.grad_clip,
            'use_scheduler': agent.use_scheduler
        }
    }
    if agent.scheduler is not None:
        checkpoint['scheduler_state'] = agent.scheduler.state_dict()
    
    # Atomic write
    import os
    temp_path = str(save_path) + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, save_path)


def load_checkpoint(checkpoint_path, model):
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load state into
        
    Returns:
        dict: Checkpoint data containing states and episode info
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    return checkpoint


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
    
    # Load from checkpoint if resuming
    start_episode = args.start_episode
    previous_rewards = []
    
    if args.resume:
        print(f"\nLoading checkpoint from: {args.resume}")
        checkpoint = load_checkpoint(args.resume, policy)
        start_episode = checkpoint['episode']
        previous_rewards = checkpoint['rewards']
        print(f"Resuming from episode {start_episode}")
        print(f"Previous training: {len(previous_rewards)} episodes completed\n")
    
    agent = REINFORCEAgent(
        policy, 
        lr=args.lr, 
        gamma=args.gamma,
        use_scheduler=(not args.resume)  # Disable scheduler if resuming
    )
    
    # Restore optimizer and scheduler state if resuming
    if args.resume:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint and agent.scheduler is not None:
            agent.scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # Create environment
    env = create_env()
    
    # Setup periodic saving callback
    all_rewards = previous_rewards.copy()
    
    def save_checkpoint_callback(episode, episode_rewards):
        """Save intermediate results every 50 episodes."""
        import json
        import os
        
        # Merge with previous rewards
        current_rewards = previous_rewards + episode_rewards
        all_rewards.clear()
        all_rewards.extend(current_rewards)
        
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
                "episode_rewards": current_rewards,
                "num_episodes": start_episode + episode
            }, f, indent=2)
        os.replace(temp_path, rewards_path)
        
        # Save model checkpoint
        checkpoint_dir = Path('checkpoints') / args.mode
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f'seed{args.seed}_episode{start_episode + episode}.pth'
        save_checkpoint(agent, start_episode + episode, current_rewards, 
                       agent.optimizer, checkpoint_path)
        
        print(f"  [Checkpoint saved: episode {start_episode + episode}]")
    
    # Train with periodic checkpointing
    print(f"Training from episode {start_episode} to {start_episode + args.episodes}...\n")
    episode_rewards = agent.train(
        env, 
        episodes=args.episodes, 
        save_callback=save_checkpoint_callback, 
        seed=args.seed,
        start_episode=start_episode
    )
    
    # Merge all rewards
    final_rewards = previous_rewards + episode_rewards
    
    # Save results
    save_training_results(final_rewards, args.seed, args.mode)
    plot_training_curve(final_rewards, args.seed, args.mode)
    print_summary(final_rewards, args.mode)
    
    # Save final model checkpoint
    checkpoint_dir = Path('checkpoints') / args.mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = checkpoint_dir / f'seed{args.seed}_final.pth'
    save_checkpoint(agent, start_episode + args.episodes, final_rewards,
                   agent.optimizer, final_checkpoint_path)
    print(f"Final checkpoint saved to {final_checkpoint_path}")


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
    
    # Load from checkpoint if resuming
    start_episode = args.start_episode
    previous_rewards = []
    
    if args.resume:
        print(f"\nLoading checkpoint from: {args.resume}")
        checkpoint = load_checkpoint(args.resume, policy)
        start_episode = checkpoint['episode']
        previous_rewards = checkpoint['rewards']
        print(f"Resuming from episode {start_episode}")
        print(f"Previous training: {len(previous_rewards)} episodes completed\n")
    
    # Create agent
    agent = REINFORCEAgent(
        policy, 
        lr=args.lr, 
        gamma=args.gamma,
        use_scheduler=(not args.resume)
    )
    
    # Restore optimizer and scheduler state if resuming
    if args.resume:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint and agent.scheduler is not None:
            agent.scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # Create environment
    env = create_env()
    
    # Setup periodic saving callback
    all_rewards = previous_rewards.copy()
    
    def save_checkpoint_callback(episode, episode_rewards):
        """Save intermediate results every 50 episodes."""
        import json
        import os
        
        # Merge with previous rewards
        current_rewards = previous_rewards + episode_rewards
        all_rewards.clear()
        all_rewards.extend(current_rewards)
        
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
                "episode_rewards": current_rewards,
                "num_episodes": start_episode + episode
            }, f, indent=2)
        os.replace(temp_path, rewards_path)
        
        # Save model checkpoint
        checkpoint_dir = Path('checkpoints') / args.mode
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f'seed{args.seed}_episode{start_episode + episode}.pth'
        save_checkpoint(agent, start_episode + episode, current_rewards,
                       agent.optimizer, checkpoint_path)
        
        print(f"  [Checkpoint saved: episode {start_episode + episode}]")
    
    # Train with periodic checkpointing
    print(f"Training from episode {start_episode} to {start_episode + args.episodes}...\n")
    if args.diff_method == 'parameter-shift':
        print("⚠️  Warning: parameter-shift gradients are 100-1000x slower than backprop")
        print("   This is expected - parameter-shift mimics real hardware constraints\n")
    
    episode_rewards = agent.train(
        env,
        episodes=args.episodes,
        save_callback=save_checkpoint_callback,
        seed=args.seed,
        start_episode=start_episode
    )
    
    # Merge all rewards
    final_rewards = previous_rewards + episode_rewards
    
    # Save results
    save_training_results(final_rewards, args.seed, args.mode)
    plot_training_curve(final_rewards, args.seed, args.mode)
    print_summary(final_rewards, args.mode)
    
    # Save final model checkpoint
    checkpoint_dir = Path('checkpoints') / args.mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = checkpoint_dir / f'seed{args.seed}_final.pth'
    save_checkpoint(agent, start_episode + args.episodes, final_rewards,
                   agent.optimizer, final_checkpoint_path)
    print(f"Final checkpoint saved to {final_checkpoint_path}")


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
