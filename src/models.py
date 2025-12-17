"""
Neural network models for CartPole policy learning.

This module contains policy network architectures:
- TinyMLP: Classical baseline (4→7→2, ~51 parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    """
    Tiny Multi-Layer Perceptron for CartPole policy.
    
    Architecture: 4 inputs → 7 hidden → 2 outputs
    Parameters: (4×7 + 7) + (7×2 + 2) = 51 total
    
    This network serves as the classical baseline with parameter count
    matched to the quantum circuit (~50 parameters).
    """
    
    def __init__(self):
        super(TinyMLP, self).__init__()
        
        # Input layer: 4 state variables → 7 hidden units
        self.fc1 = nn.Linear(4, 7)
        
        # Output layer: 7 hidden → 2 actions
        self.fc2 = nn.Linear(7, 2)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): CartPole state, shape (4,) or (batch, 4)
            
        Returns:
            torch.Tensor: Action probabilities, shape (2,) or (batch, 2)
        """
        # Hidden layer with ReLU activation
        hidden = F.relu(self.fc1(state))
        
        # Output layer with softmax for action probabilities
        logits = self.fc2(hidden)
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def get_action(self, state):
        """
        Sample an action from the policy distribution.
        
        Args:
            state (torch.Tensor): CartPole state, shape (4,)
            
        Returns:
            tuple: (action, log_prob)
                - action (int): Sampled action (0 or 1)
                - log_prob (torch.Tensor): Log probability of the action
        """
        # Get action probabilities
        probs = self.forward(state)
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def get_log_prob(self, state, action):
        """
        Compute log probability of a specific action.
        
        Used during policy gradient updates.
        
        Args:
            state (torch.Tensor): State, shape (4,) or (batch, 4)
            action (torch.Tensor): Action, shape () or (batch,)
            
        Returns:
            torch.Tensor: Log probability of the action
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)
    
    def count_parameters(self):
        """
        Count total trainable parameters.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
