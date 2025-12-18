"""
Agent visualization script for CartPole demos.

Loads trained policy and records agent playing CartPole.
Saves as PNG sequence or GIF for demonstration.
"""

import torch
import gymnasium as gym
import numpy as np
from pathlib import Path
import argparse
from PIL import Image

from src.models import TinyMLP, QuantumPolicy


def load_model(checkpoint_path, model_type='classical'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_type: 'classical' or 'quantum'
        
    Returns:
        Loaded model in eval mode
    """
    if model_type == 'classical':
        model = TinyMLP()
    else:
        model = QuantumPolicy(n_qubits=4, n_layers=3, measurement='softmax')
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    return model


def record_episode(env, model, max_steps=500):
    """
    Record one episode of agent playing.
    
    Args:
        env: Gymnasium environment
        model: Trained policy
        max_steps: Maximum steps per episode
        
    Returns:
        list: RGB frames from episode
    """
    frames = []
    state, _ = env.reset()
    
    for _ in range(max_steps):
        # Render frame
        frame = env.render()
        frames.append(Image.fromarray(frame))
        
        # Get action
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = model(state_tensor)
            action = torch.argmax(action_probs).item()
        
        # Step environment
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if done:
            break
    
    return frames


def save_as_gif(frames, output_path, duration=50):
    """Save frames as animated GIF."""
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )


def main():
    parser = argparse.ArgumentParser(description='Visualize trained CartPole agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['classical', 'quantum'],
                       default='classical', help='Type of model')
    parser.add_argument('--output', type=str, default='agent_demo.gif',
                       help='Output path for demo GIF')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to record')
    
    args = parser.parse_args()
    
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model_type)
    
    print(f"Recording {args.episodes} episodes...")
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    all_frames = []
    for ep in range(args.episodes):
        print(f"  Episode {ep + 1}/{args.episodes}...")
        frames = record_episode(env, model)
        all_frames.extend(frames)
        
        # Add separator frames
        if ep < args.episodes - 1:
            all_frames.extend([frames[-1]] * 10)  # Pause between episodes
    
    env.close()
    
    print(f"Saving GIF to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_as_gif(all_frames, args.output)
    
    print(f"✓ Demo saved ({len(all_frames)} frames)")


if __name__ == '__main__':
    main()
