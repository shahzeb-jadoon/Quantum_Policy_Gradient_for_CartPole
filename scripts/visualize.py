"""
Agent visualization script for CartPole demos.

Loads trained policy and records agent playing CartPole.
Supports extended episode duration and HUD overlay for step counting.
"""

import torch
import gymnasium as gym
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont

from src.models import TinyMLP, QuantumPolicy
from src.env_wrapper import normalize_state


def overlay_text(frame, text, position=(10, 10), color=(0, 0, 255)):
    """
    Draw text overlay on frame.
    
    Args:
        frame: numpy array (H, W, 3)
        text: string to display
        position: (x, y) tuple for text position
        color: RGB tuple for text color
        
    Returns:
        numpy array with text overlay
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Try to use a better font if available, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text(position, text, fill=color, font=font)
    
    return np.array(img)


def load_model(checkpoint_path, model_type='classical'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_type: 'classical' or 'quantum'
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Handle both old format (direct state_dict) and new format (dict with 'model_state_dict')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        n_qubits = checkpoint.get('n_qubits', 4)
        n_layers = checkpoint.get('n_layers', 3)
        diff_method = checkpoint.get('diff_method', 'backprop')
    else:
        # Old format - checkpoint IS the state_dict
        state_dict = checkpoint
        n_qubits = 4
        n_layers = 3
        diff_method = 'backprop'
    
    if model_type == 'classical':
        model = TinyMLP()
        model.load_state_dict(state_dict)
    else:
        # Load quantum model with configuration
        model = QuantumPolicy(
            n_qubits=n_qubits,
            n_layers=n_layers,
            measurement='softmax',
            diff_method=diff_method
        )
        model.load_state_dict(state_dict)
    
    model.eval()
    return model


def record_episode(env, model, episode_num=1, max_steps=5000, show_hud=True):
    """
    Record one episode of agent playing.
    
    Args:
        env: Gymnasium environment
        model: Trained policy
        episode_num: Episode number for HUD
        max_steps: Maximum steps per episode
        show_hud: Whether to show step counter overlay
        
    Returns:
        list: RGB frames from episode
    """
    frames = []
    state, _ = env.reset()
    step_count = 0
    
    for step in range(max_steps):
        # Render frame
        frame = env.render()
        
        # Add HUD overlay
        if show_hud:
            info_text = f"Episode: {episode_num} | Step: {step + 1}/{max_steps}"
            frame = overlay_text(frame, info_text)
        
        frames.append(Image.fromarray(frame))
        
        # Normalize state and get action
        state_norm = normalize_state(state)
        state_tensor = torch.FloatTensor(state_norm)
        
        with torch.no_grad():
            action_probs = model(state_tensor)
            action = torch.argmax(action_probs).item()
        
        # Step environment
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count = step + 1
        
        if done:
            break
    
    return frames, step_count


def save_as_gif(frames, output_path, duration=50):
    """
    Save frames as animated GIF.
    
    Args:
        frames: List of PIL Images
        output_path: Path to save GIF
        duration: Milliseconds per frame
    """
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
    parser.add_argument('--duration', type=int, default=None,
                       help='Target duration in seconds (overrides default 500-step limit)')
    parser.add_argument('--no-hud', action='store_true',
                       help='Disable step counter overlay')
    
    args = parser.parse_args()
    
    # Calculate max steps based on duration if specified
    # CartPole runs at 50 Hz (0.02s per step)
    if args.duration is not None:
        max_steps = int(args.duration / 0.02)
        print(f"🎥 Recording for {args.duration}s ({max_steps} steps per episode)")
    else:
        max_steps = 500  # Default CartPole limit
    
    # Create environment with extended limit if needed
    env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=max_steps)
    
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model_type)
    
    print(f"Recording {args.episodes} episodes...")
    
    all_frames = []
    total_steps = 0
    
    for ep in range(args.episodes):
        print(f"  Episode {ep + 1}/{args.episodes}...")
        
        # Add episode start marker
        if ep > 0 and not args.no_hud:
            # Create a marker frame
            marker_frame = env.render()
            marker_text = f"*** EPISODE {ep + 1} START ***"
            marker_frame = overlay_text(marker_frame, marker_text, position=(150, 200), color=(255, 0, 0))
            all_frames.extend([Image.fromarray(marker_frame)] * 10)  # Flash for 0.5 seconds
        
        frames, step_count = record_episode(
            env, model, 
            episode_num=ep + 1, 
            max_steps=max_steps,
            show_hud=not args.no_hud
        )
        all_frames.extend(frames)
        total_steps += step_count
        
        print(f"    Completed {step_count} steps")
        
        # Add separator frames between episodes
        if ep < args.episodes - 1:
            all_frames.extend([frames[-1]] * 10)  # Pause between episodes
    
    env.close()
    
    print(f"Saving GIF to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_as_gif(all_frames, args.output)
    
    print(f"✓ Demo saved ({len(all_frames)} frames, {total_steps} total steps)")


if __name__ == '__main__':
    main()
