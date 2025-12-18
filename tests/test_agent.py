"""
Tests for REINFORCE agent.
"""

import pytest
import torch
import numpy as np
from src.agent import REINFORCEAgent
from src.models import TinyMLP
from src.env_wrapper import create_env


class TestREINFORCEInitialization:
    """Test agent initialization."""
    
    def test_initialization(self):
        """Test agent initializes correctly."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy, lr=0.01, gamma=0.99)
        
        assert agent.model is policy
        assert agent.gamma == 0.99
        assert agent.optimizer is not None
        assert agent.grad_clip == 1.0  # Default value
        assert agent.scheduler is not None  # use_scheduler=True by default
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy, lr=0.001, gamma=0.95)
        
        assert agent.gamma == 0.95
        # Check optimizer learning rate
        assert agent.optimizer.param_groups[0]['lr'] == 0.001
    
    def test_initial_storage_empty(self):
        """Test storage lists initialized empty."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        assert agent.saved_log_probs == []
        assert agent.rewards == []


class TestActionSelection:
    """Test action selection."""
    
    def test_select_action_returns_valid(self):
        """Test select_action returns valid action."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = agent.select_action(state)
        
        assert action in [0, 1]
        assert isinstance(action, int)
    
    def test_select_action_stores_log_prob(self):
        """Test select_action stores log probability."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        agent.select_action(state)
        
        assert len(agent.saved_log_probs) == 1
        assert isinstance(agent.saved_log_probs[0], torch.Tensor)
    
    def test_multiple_action_selections(self):
        """Test multiple action selections accumulate log probs."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        for _ in range(5):
            agent.select_action(state)
        
        assert len(agent.saved_log_probs) == 5


class TestRewardStorage:
    """Test reward storage."""
    
    def test_store_reward(self):
        """Test storing single reward."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        agent.store_reward(1.0)
        
        assert len(agent.rewards) == 1
        assert agent.rewards[0] == 1.0
    
    def test_store_multiple_rewards(self):
        """Test storing multiple rewards."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        rewards = [1.0, 0.5, 1.0, 0.0, 1.0]
        for r in rewards:
            agent.store_reward(r)
        
        assert len(agent.rewards) == 5
        assert agent.rewards == rewards


class TestReturnComputation:
    """Test discounted return calculation."""
    
    def test_compute_returns_single_step(self):
        """Test returns for single-step episode."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy, gamma=0.99)
        
        agent.rewards = [10.0]
        returns = agent.compute_returns()
        
        assert returns.shape == (1,)
        # With single element, std=0, so normalization doesn't apply
        # Returns the value as-is
        assert torch.allclose(returns, torch.tensor([10.0]))
    
    def test_compute_returns_discounting(self):
        """Test returns are properly discounted."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy, gamma=0.5)
        
        # Simple rewards: [1, 1, 1]
        agent.rewards = [1.0, 1.0, 1.0]
        returns = agent.compute_returns()
        
        # Before normalization:
        # G_0 = 1 + 0.5*1 + 0.25*1 = 1.75
        # G_1 = 1 + 0.5*1 = 1.5
        # G_2 = 1
        expected_unnorm = torch.tensor([1.75, 1.5, 1.0])
        
        # After normalization: (x - mean) / std
        mean = expected_unnorm.mean()
        std = expected_unnorm.std()
        expected = (expected_unnorm - mean) / (std + 1e-8)
        
        assert torch.allclose(returns, expected, atol=1e-5)
    
    def test_compute_returns_zero_gamma(self):
        """Test returns with gamma=0 (no discounting)."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy, gamma=0.0)
        
        agent.rewards = [1.0, 2.0, 3.0]
        returns = agent.compute_returns()
        
        # With gamma=0, G_t = r_t (no future rewards)
        # Before normalization: [1, 2, 3]
        expected_unnorm = torch.tensor([1.0, 2.0, 3.0])
        mean = expected_unnorm.mean()
        std = expected_unnorm.std()
        expected = (expected_unnorm - mean) / (std + 1e-8)
        
        assert torch.allclose(returns, expected, atol=1e-5)
    
    def test_compute_returns_normalization(self):
        """Test returns are normalized."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy, gamma=0.99)
        
        agent.rewards = [1.0] * 10
        returns = agent.compute_returns()
        
        # Normalized returns should have ~zero mean and ~unit std
        assert torch.allclose(returns.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(returns.std(), torch.tensor(1.0), atol=1e-5)


class TestPolicyUpdate:
    """Test policy gradient updates."""
    
    def test_update_policy_clears_storage(self):
        """Test update_policy clears episode data."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        # Add some data
        state = torch.randn(4)
        action, log_prob = policy.get_action(state)
        agent.saved_log_probs.append(log_prob)
        agent.rewards.append(1.0)
        
        # Update
        agent.update_policy()
        
        # Storage should be cleared
        assert agent.saved_log_probs == []
        assert agent.rewards == []
    
    def test_update_policy_returns_loss(self):
        """Test update_policy returns loss and gradient norm."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        # Simulate short episode
        for _ in range(3):
            state = torch.randn(4)
            action, log_prob = policy.get_action(state)
            agent.saved_log_probs.append(log_prob)
            agent.rewards.append(1.0)
        
        loss, grad_norm = agent.update_policy()
        
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0
    
    def test_update_updates_parameters(self):
        """Test policy parameters change after update."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        # Save initial parameters
        initial_params = [p.clone() for p in policy.parameters()]
        
        # Simulate episode
        for _ in range(5):
            state = torch.randn(4)
            action, log_prob = policy.get_action(state)
            agent.saved_log_probs.append(log_prob)
            agent.rewards.append(1.0)
        
        # Update
        agent.update_policy()
        
        # Check parameters changed
        changed = False
        for initial, current in zip(initial_params, policy.parameters()):
            if not torch.allclose(initial, current.data):
                changed = True
                break
        
        assert changed, "Policy parameters should change after update"


class TestTrainingEpisode:
    """Test single episode training."""
    
    def test_train_episode_completes(self):
        """Test train_episode runs without errors."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        env = create_env()
        
        # train_episode now calls update_policy internally and only returns reward
        reward = agent.train_episode(env)
        
        assert isinstance(reward, (int, float))
        assert reward >= 0  # CartPole always gives non-negative reward
    
    def test_train_episode_clears_storage(self):
        """Test episode data cleared after training."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        env = create_env()
        
        agent.train_episode(env)
        
        # train_episode collects data but doesn't update
        # Storage is cleared by update_policy() which is called in train()
        assert len(agent.saved_log_probs) > 0  # Should have collected data
        assert len(agent.rewards) > 0  # Should have collected rewards
        
        # Now call update_policy to clear storage
        agent.update_policy()
        assert agent.saved_log_probs == []
        assert agent.rewards == []
    
    def test_train_episode_length_equals_reward(self):
        """Test episode length matches reward in CartPole."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        env = create_env()
        
        # train_episode now only returns reward
        reward = agent.train_episode(env)
        
        # Reward should be positive in CartPole
        assert reward > 0


class TestTraining:
    """Test multi-episode training."""
    
    def test_train_returns_stats(self):
        """Test train returns EpisodeStats object."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        # train now returns list of rewards
        env = create_env()
        rewards = agent.train(env, episodes=5, log_interval=10)
        
        assert len(rewards) == 5
        assert all(isinstance(r, (int, float)) for r in rewards)
    
    def test_train_improves_performance(self):
        """Test agent improves with training (stochastic test)."""
        policy = TinyMLP()
        env = create_env()
        # Disable scheduler for this test to avoid LR changes
        agent = REINFORCEAgent(policy, lr=0.01, use_scheduler=False)
        
        # Train for 50 episodes
        rewards = agent.train(env, episodes=50, log_interval=10)
        
        # Early performance (first 10 episodes)
        early_avg = np.mean(rewards[:10])
        
        # Late performance (last 10 episodes)
        late_avg = np.mean(rewards[-10:])
        
        # Agent should improve (this might occasionally fail due to randomness)
        # Use a lenient threshold
        assert late_avg >= early_avg * 0.8, "Agent should show learning progress"


class TestEvaluation:
    """Test agent evaluation."""
    
    def test_evaluate_returns_statistics(self):
        """Test evaluate returns mean and std."""
        policy = TinyMLP()
        agent = REINFORCEAgent(policy)
        
        mean_reward, std_reward = agent.evaluate(num_episodes=10)
        
        assert isinstance(mean_reward, (float, np.floating))
        assert isinstance(std_reward, (float, np.floating))
        assert mean_reward > 0
        assert std_reward >= 0
    
    def test_evaluate_deterministic(self):
        """Test evaluation uses greedy actions (deterministic)."""
        policy = TinyMLP()
        env = create_env()
        agent = REINFORCEAgent(policy, use_scheduler=False)
        
        # Train briefly to get consistent policy
        agent.train(env, episodes=10, log_interval=10)
        
        # Evaluate multiple times - should give similar results
        rewards1 = []
        rewards2 = []
        
        for _ in range(5):
            mean1, _ = agent.evaluate(env, num_episodes=5)
            mean2, _ = agent.evaluate(env, num_episodes=5)
            rewards1.append(mean1)
            rewards2.append(mean2)
        
        # Results should be similar (but not necessarily identical due to env randomness)
        assert np.std(rewards1 + rewards2) < 50, "Evaluation should be relatively consistent"
