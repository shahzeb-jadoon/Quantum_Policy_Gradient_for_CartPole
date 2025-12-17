"""
CartPole environment wrapper with utilities for state normalization and monitoring.

This module provides wrapper functions for the CartPole-v1 environment,
including state normalization and episode statistics tracking.
"""

import gymnasium as gym
import numpy as np


def create_env(render_mode=None):
    """
    Create and initialize CartPole-v1 environment.
    
    Args:
        render_mode (str, optional): Rendering mode ('human', 'rgb_array', or None)
        
    Returns:
        gym.Env: Initialized CartPole-v1 environment
    """
    env = gym.make('CartPole-v1', render_mode=render_mode)
    return env


def normalize_state(state):
    """
    Normalize CartPole state variables to standard ranges.
    
    CartPole state: [x, x_dot, theta, theta_dot]
    - x: cart position (-4.8 to 4.8)
    - x_dot: cart velocity
    - theta: pole angle (-0.418 to 0.418 radians, ~24 degrees)
    - theta_dot: pole angular velocity
    
    Args:
        state (np.ndarray): Raw state from environment, shape (4,)
        
    Returns:
        np.ndarray: Normalized state, shape (4,)
    """
    # Normalization bounds based on CartPole-v1 specifications
    state_bounds = np.array([
        4.8,    # x position bound
        np.inf,  # x velocity (unbounded, use as-is)
        0.418,  # theta bound (radians)
        np.inf   # theta velocity (unbounded, use as-is)
    ])
    
    # Normalize bounded variables, keep unbounded as-is
    normalized = np.zeros_like(state)
    for i in range(len(state)):
        if state_bounds[i] != np.inf:
            normalized[i] = state[i] / state_bounds[i]
        else:
            # For unbounded variables, apply tanh for soft clipping
            normalized[i] = np.tanh(state[i] / 2.0)
    
    return normalized


class EpisodeStats:
    """
    Track statistics for CartPole episodes during training.
    
    Attributes:
        episode_rewards (list): Total reward per episode
        episode_lengths (list): Number of steps per episode
        current_episode_reward (float): Cumulative reward in current episode
        current_episode_length (int): Steps in current episode
    """
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def step(self, reward):
        """
        Update statistics for a single step.
        
        Args:
            reward (float): Reward received from environment
        """
        self.current_episode_reward += reward
        self.current_episode_length += 1
    
    def end_episode(self):
        """
        Record episode statistics and reset counters for next episode.
        
        Returns:
            tuple: (total_reward, episode_length)
        """
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        
        reward = self.current_episode_reward
        length = self.current_episode_length
        
        # Reset for next episode
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        return reward, length
    
    def get_recent_average(self, n=100):
        """
        Calculate average reward over last n episodes.
        
        Args:
            n (int): Number of recent episodes to average
            
        Returns:
            float: Average reward, or 0.0 if no episodes recorded
        """
        if len(self.episode_rewards) == 0:
            return 0.0
        recent = self.episode_rewards[-n:]
        return np.mean(recent)
    
    def is_solved(self, threshold=195.0, n=100):
        """
        Check if CartPole is solved (avg reward > threshold over n episodes).
        
        Args:
            threshold (float): Reward threshold for "solved" status
            n (int): Number of episodes to average
            
        Returns:
            bool: True if environment is solved
        """
        if len(self.episode_rewards) < n:
            return False
        return self.get_recent_average(n) >= threshold
