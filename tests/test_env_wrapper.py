"""
Tests for CartPole environment wrapper utilities.
"""

import pytest
import numpy as np
from src.env_wrapper import create_env, normalize_state, EpisodeStats


class TestEnvironmentCreation:
    """Test environment initialization."""
    
    def test_create_env_default(self):
        """Test creating environment with default settings."""
        env = create_env()
        assert env is not None
        assert env.spec.id == 'CartPole-v1'
    
    def test_create_env_with_render(self):
        """Test creating environment with rendering mode."""
        env = create_env(render_mode='rgb_array')
        assert env is not None
        assert env.render_mode == 'rgb_array'
    
    def test_env_reset(self):
        """Test environment reset returns correct state shape."""
        env = create_env()
        state, info = env.reset()
        assert state.shape == (4,), f"Expected state shape (4,), got {state.shape}"
        assert isinstance(info, dict)
    
    def test_env_step(self):
        """Test environment step with valid action."""
        env = create_env()
        state, _ = env.reset()
        
        # Take action 0 (push left)
        next_state, reward, terminated, truncated, info = env.step(0)
        
        assert next_state.shape == (4,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestStateNormalization:
    """Test state normalization function."""
    
    def test_normalize_state_shape(self):
        """Test that normalization preserves state shape."""
        state = np.array([1.0, 0.5, 0.1, -0.3])
        normalized = normalize_state(state)
        assert normalized.shape == state.shape
    
    def test_normalize_bounded_variables(self):
        """Test normalization of bounded state variables (x, theta)."""
        # State at bounds: [x=4.8, x_dot=0, theta=0.418, theta_dot=0]
        state = np.array([4.8, 0.0, 0.418, 0.0])
        normalized = normalize_state(state)
        
        # x should be normalized to 1.0
        assert np.isclose(normalized[0], 1.0, atol=1e-6)
        # theta should be normalized to 1.0
        assert np.isclose(normalized[2], 1.0, atol=1e-6)
    
    def test_normalize_zero_state(self):
        """Test normalization of zero state."""
        state = np.zeros(4)
        normalized = normalize_state(state)
        assert np.allclose(normalized, np.zeros(4))
    
    def test_normalize_unbounded_variables(self):
        """Test that unbounded variables (velocities) use tanh."""
        # Large velocity should be soft-clipped by tanh
        state = np.array([0.0, 10.0, 0.0, 10.0])
        normalized = normalize_state(state)
        
        # Tanh(10/2) = tanh(5) ≈ 0.9999
        assert -1.0 <= normalized[1] <= 1.0
        assert -1.0 <= normalized[3] <= 1.0
        assert normalized[1] > 0.99  # Should be close to 1 for large values


class TestEpisodeStats:
    """Test episode statistics tracking."""
    
    def test_initialization(self):
        """Test EpisodeStats initializes with empty lists."""
        stats = EpisodeStats()
        assert stats.episode_rewards == []
        assert stats.episode_lengths == []
        assert stats.current_episode_reward == 0.0
        assert stats.current_episode_length == 0
    
    def test_step_accumulation(self):
        """Test that step() correctly accumulates rewards."""
        stats = EpisodeStats()
        
        stats.step(1.0)
        assert stats.current_episode_reward == 1.0
        assert stats.current_episode_length == 1
        
        stats.step(1.0)
        assert stats.current_episode_reward == 2.0
        assert stats.current_episode_length == 2
    
    def test_end_episode(self):
        """Test that end_episode() records stats and resets counters."""
        stats = EpisodeStats()
        
        # Simulate 10-step episode with reward=1 per step
        for _ in range(10):
            stats.step(1.0)
        
        reward, length = stats.end_episode()
        
        assert reward == 10.0
        assert length == 10
        assert len(stats.episode_rewards) == 1
        assert len(stats.episode_lengths) == 1
        assert stats.current_episode_reward == 0.0
        assert stats.current_episode_length == 0
    
    def test_multiple_episodes(self):
        """Test tracking multiple episodes."""
        stats = EpisodeStats()
        
        # Episode 1: 5 steps
        for _ in range(5):
            stats.step(1.0)
        stats.end_episode()
        
        # Episode 2: 10 steps
        for _ in range(10):
            stats.step(1.0)
        stats.end_episode()
        
        assert len(stats.episode_rewards) == 2
        assert stats.episode_rewards == [5.0, 10.0]
        assert stats.episode_lengths == [5, 10]
    
    def test_get_recent_average_empty(self):
        """Test average with no episodes."""
        stats = EpisodeStats()
        assert stats.get_recent_average() == 0.0
    
    def test_get_recent_average_single(self):
        """Test average with single episode."""
        stats = EpisodeStats()
        for _ in range(10):
            stats.step(1.0)
        stats.end_episode()
        
        assert stats.get_recent_average(n=100) == 10.0
    
    def test_get_recent_average_windowed(self):
        """Test average over n most recent episodes."""
        stats = EpisodeStats()
        
        # Create 150 episodes with increasing rewards
        for ep in range(150):
            for _ in range(ep + 1):  # Episode reward = episode number + 1
                stats.step(1.0)
            stats.end_episode()
        
        # Last 100 episodes: rewards from 51 to 150
        # Average = (51 + 150) / 2 = 100.5
        avg = stats.get_recent_average(n=100)
        expected = np.mean(list(range(51, 151)))
        assert np.isclose(avg, expected, atol=0.1)
    
    def test_is_solved_insufficient_episodes(self):
        """Test is_solved() returns False with <100 episodes."""
        stats = EpisodeStats()
        
        # Only 50 episodes with perfect score
        for _ in range(50):
            for _ in range(200):
                stats.step(1.0)
            stats.end_episode()
        
        assert stats.is_solved() is False
    
    def test_is_solved_below_threshold(self):
        """Test is_solved() returns False when avg < threshold."""
        stats = EpisodeStats()
        
        # 100 episodes with reward=100 each (below 195)
        for _ in range(100):
            for _ in range(100):
                stats.step(1.0)
            stats.end_episode()
        
        assert stats.is_solved(threshold=195.0) == False
    
    def test_is_solved_above_threshold(self):
        """Test is_solved() returns True when avg >= threshold."""
        stats = EpisodeStats()
        
        # 100 episodes with reward=200 each (above 195)
        for _ in range(100):
            for _ in range(200):
                stats.step(1.0)
            stats.end_episode()
        
        assert stats.is_solved(threshold=195.0) == True
    
    def test_is_solved_custom_parameters(self):
        """Test is_solved() with custom threshold and window."""
        stats = EpisodeStats()
        
        # 50 episodes with reward=150 each
        for _ in range(50):
            for _ in range(150):
                stats.step(1.0)
            stats.end_episode()
        
        # Should be solved with threshold=100 and n=50
        assert stats.is_solved(threshold=100.0, n=50) == True
        # Should not be solved with threshold=200
        assert stats.is_solved(threshold=200.0, n=50) == False
