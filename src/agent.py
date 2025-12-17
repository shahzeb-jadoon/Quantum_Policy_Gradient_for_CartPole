"""
REINFORCE agent for policy gradient reinforcement learning.

This module implements the Monte Carlo Policy Gradient algorithm (REINFORCE)
for training policy networks on CartPole.
"""

import torch
import torch.optim as optim
import numpy as np
from src.env_wrapper import create_env, normalize_state, EpisodeStats


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    
    Trains a policy network by:
    1. Rolling out full episodes
    2. Computing discounted returns
    3. Updating policy to increase probability of high-reward actions
    
    Mathematical formulation:
        ∇J(θ) = E[∇log π(a|s) * G_t]
        where G_t is the return from timestep t
    """
    
    def __init__(self, policy, lr=0.01, gamma=0.99):
        """
        Initialize REINFORCE agent.
        
        Args:
            policy: Policy network (e.g., TinyMLP)
            lr (float): Learning rate for optimizer
            gamma (float): Discount factor for returns
        """
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        
        # Storage for episode data
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """
        Select action from policy and save log probability.
        
        Args:
            state (np.ndarray): Environment state
            
        Returns:
            int: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get action from policy
        action, log_prob = self.policy.get_action(state_tensor)
        
        # Save log probability for later update
        self.saved_log_probs.append(log_prob)
        
        return action
    
    def store_reward(self, reward):
        """
        Store reward for current timestep.
        
        Args:
            reward (float): Reward from environment
        """
        self.rewards.append(reward)
    
    def compute_returns(self):
        """
        Compute discounted returns for the episode.
        
        Returns are calculated backwards from the end of the episode:
            G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        
        Returns:
            torch.Tensor: Discounted returns for each timestep
        """
        returns = []
        G = 0
        
        # Calculate returns backwards (from end to start)
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns for stability (zero mean, unit variance)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self):
        """
        Update policy using collected episode data.
        
        Implements the REINFORCE update:
            loss = -Σ log π(a_t|s_t) * G_t
        
        Returns:
            float: Policy loss value
        """
        # Compute returns
        returns = self.compute_returns()
        
        # Compute policy loss
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            # Negative because we want to maximize reward (minimize negative reward)
            policy_loss.append(-log_prob * G)
        
        # Sum losses across all timesteps
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def train_episode(self, env):
        """
        Run one full training episode.
        
        Args:
            env: Gymnasium environment
            
        Returns:
            tuple: (total_reward, episode_length, loss)
        """
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        
        # Rollout episode
        while True:
            # Normalize state
            state_norm = normalize_state(state)
            
            # Select action
            action = self.select_action(state_norm)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store reward
            self.store_reward(reward)
            
            total_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        # Update policy at end of episode
        loss = self.update_policy()
        
        return total_reward, episode_length, loss
    
    def train(self, num_episodes, env=None, verbose=True):
        """
        Train agent for multiple episodes.
        
        Args:
            num_episodes (int): Number of episodes to train
            env: Environment (created if None)
            verbose (bool): Print progress
            
        Returns:
            EpisodeStats: Training statistics
        """
        if env is None:
            env = create_env()
        
        stats = EpisodeStats()
        
        for episode in range(num_episodes):
            # Train one episode
            reward, length, loss = self.train_episode(env)
            
            # Update statistics
            stats.current_episode_reward = reward
            stats.current_episode_length = length
            stats.end_episode()
            
            # Print progress
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = stats.get_recent_average(n=50)
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward (50 ep): {avg_reward:.1f} | "
                      f"Loss: {loss:.4f}")
        
        return stats
    
    def evaluate(self, env=None, num_episodes=100):
        """
        Evaluate agent without training.
        
        Args:
            env: Environment (created if None)
            num_episodes (int): Number of evaluation episodes
            
        Returns:
            tuple: (mean_reward, std_reward)
        """
        if env is None:
            env = create_env()
        
        rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            while True:
                state_norm = normalize_state(state)
                state_tensor = torch.FloatTensor(state_norm)
                
                # Greedy action (no exploration)
                with torch.no_grad():
                    probs = self.policy(state_tensor)
                    action = torch.argmax(probs).item()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            rewards.append(total_reward)
        
        return np.mean(rewards), np.std(rewards)
