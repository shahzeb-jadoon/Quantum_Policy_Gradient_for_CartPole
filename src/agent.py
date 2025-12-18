"""
REINFORCE agent for policy gradient training.

Implements vanilla policy gradient algorithm with:
- Return normalization for training stability
- Gradient clipping to prevent explosions
- Barren plateau detection for quantum circuits
- Adaptive learning rate scheduling
- Progress monitoring with tqdm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from tqdm import tqdm
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
    
    def __init__(self, model, lr=0.01, gamma=0.99, grad_clip=1.0, 
                 use_scheduler=True):
        """
        Initialize REINFORCE agent.
        
        Args:
            model: Policy network (TinyMLP or QuantumPolicy)
            lr: Learning rate for optimizer
            gamma: Discount factor for returns
            grad_clip: Maximum gradient norm (None to disable)
            use_scheduler: Whether to use adaptive learning rate
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.lr = lr
        self.grad_clip = grad_clip
        self.use_scheduler = use_scheduler
        
        # Adaptive learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=50,
                threshold=1e-4
            )
        else:
            self.scheduler = None
        
        # Episode history buffers
        self.saved_log_probs = []
        self.rewards = []
        
        # Gradient monitoring
        self.grad_norms = []
    
    def select_action(self, state):
        """
        Select action from current policy.
        
        Args:
            state (np.ndarray): Environment state
            
        Returns:
            int: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get action from policy
        action, log_prob = self.model.get_action(state_tensor)
        
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
    
    def get_gradient_norm(self):
        """
        Compute L2 norm of model gradients.
        
        Returns:
            float: Total gradient norm
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def check_barren_plateau(self, grad_norm, threshold=1e-6):
        """
        Detect barren plateau (vanishing gradients).
        
        Args:
            grad_norm: Current gradient norm
            threshold: Minimum acceptable gradient norm
            
        Returns:
            tuple: (is_plateau, warning_message)
        """
        if grad_norm < threshold:
            msg = f"Barren Plateau detected: gradient norm {grad_norm:.2e} < {threshold:.2e}"
            return True, msg
        return False, None
    
    def update_policy(self):
        """
        Update policy using REINFORCE algorithm.
        
        Includes gradient clipping and barren plateau detection.
        
        Returns:
            tuple: (loss, grad_norm)
        """
        # Compute returns
        returns = self.compute_returns()
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Combine into single loss
        loss = torch.stack(policy_loss).sum()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.grad_clip
            )
        
        # Monitor gradient norm
        grad_norm = self.get_gradient_norm()
        self.grad_norms.append(grad_norm)
        
        # Check for barren plateau
        is_plateau, warning = self.check_barren_plateau(grad_norm)
        if is_plateau:
            warnings.warn(warning)
        
        self.optimizer.step()
        
        # Clear episode history
        del self.saved_log_probs[:]
        del self.rewards[:]
        
        return loss.item(), grad_norm
    
    def train_episode(self, env, seed=None):
        """
        Run one full training episode.
        
        Args:
            env: Gymnasium environment
            seed: Optional random seed for environment reset
            
        Returns:
            float: total_reward for the episode
        """
        if seed is not None:
            state, _ = env.reset(seed=seed)
        else:
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
        
        return total_reward
    
    def train(self, env, episodes=500, log_interval=50, 
              save_callback=None, seed=None, start_episode=0):
        """
        Train agent for specified number of episodes.
        
        Args:
            env: Gymnasium environment
            episodes: Number of training episodes
            log_interval: Episodes between status updates
            save_callback: Optional callback for periodic saving
            seed: Base random seed for reproducibility
            start_episode: Episode number to resume from
        
        Returns:
            list: Episode rewards
        """
        episode_rewards = []
        
        # Progress bar
        pbar = tqdm(
            range(start_episode, episodes),
            desc="Training",
            unit="ep",
            initial=start_episode,
            total=episodes
        )
        
        for episode in pbar:
            # Deterministic episode seeding
            episode_seed = None if seed is None else seed + episode
            
            # Run episode
            episode_reward = self.train_episode(env, seed=episode_seed)
            episode_rewards.append(episode_reward)
            
            # Update policy
            loss, grad_norm = self.update_policy()
            
            # Update learning rate if using scheduler
            if self.scheduler is not None:
                avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
                self.scheduler.step(avg_reward)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            if len(episode_rewards) >= 50:
                recent_avg = sum(episode_rewards[-50:]) / 50
                pbar.set_postfix({
                    'reward': f'{episode_reward:.1f}',
                    'avg_50': f'{recent_avg:.1f}',
                    'loss': f'{loss:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
            
            # Periodic logging
            if (episode + 1) % log_interval == 0:
                avg_reward = sum(episode_rewards[-log_interval:]) / log_interval
                print(f"\nEpisode {episode + 1}/{episodes} | "
                      f"Avg Reward ({log_interval} ep): {avg_reward:.1f} | "
                      f"Loss: {loss:.4f} | "
                      f"Grad Norm: {grad_norm:.4f} | "
                      f"LR: {current_lr:.6f}")
                # Periodic checkpoint saving
                if save_callback is not None:
                    save_callback(episode + 1, episode_rewards)
        
        pbar.close()
        return episode_rewards
    
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
                    probs = self.model(state_tensor)
                    action = torch.argmax(probs).item()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            rewards.append(total_reward)
        
        return np.mean(rewards), np.std(rewards)
