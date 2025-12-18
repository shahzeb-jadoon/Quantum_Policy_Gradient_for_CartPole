"""
Tests for Phase 3 & 4 features added by Friend 2.

Tests cover:
- Gradient clipping
- Barren plateau detection
- Adaptive learning rate
- Training continuation (checkpoints)
- Variance metrics
- Statistical tests
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

from src.agent import REINFORCEAgent
from src.models import TinyMLP, QuantumPolicy
from src.env_wrapper import create_env, normalize_state
from src.utils import compute_variance_metrics, compute_stability_score


class TestGradientClipping:
    """Test gradient clipping functionality."""
    
    def test_gradient_clip_enabled(self):
        """Test that gradient clipping is enabled when specified."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01, grad_clip=1.0)
        assert agent.grad_clip == 1.0
    
    def test_gradient_clip_disabled(self):
        """Test that gradient clipping can be disabled."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01, grad_clip=None)
        assert agent.grad_clip is None
    
    def test_gradient_clipping_works(self):
        """Test that gradient norms are actually clipped."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01, grad_clip=0.5)
        
        # Run episode to generate gradients
        env = create_env()
        state, _ = env.reset(seed=42)
        for _ in range(10):
            action = agent.select_action(normalize_state(state))
            state, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            if terminated or truncated:
                break
        
        # Update and check gradient norm
        loss, grad_norm = agent.update_policy()
        
        # Grad norm should be at or below clip value (with small tolerance)
        assert grad_norm <= agent.grad_clip + 0.01


class TestBarrenPlateauDetection:
    """Test barren plateau detection."""
    
    def test_check_barren_plateau_exists(self):
        """Test that barren plateau detection method exists."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01)
        assert hasattr(agent, 'check_barren_plateau')
    
    def test_detects_vanishing_gradient(self):
        """Test detection of vanishing gradients."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01)
        
        # Very small gradient (barren plateau)
        is_plateau, msg = agent.check_barren_plateau(1e-8, threshold=1e-6)
        assert is_plateau is True
        assert msg is not None
        assert 'Barren Plateau' in msg
    
    def test_no_false_positive(self):
        """Test that normal gradients don't trigger detection."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01)
        
        # Normal gradient
        is_plateau, msg = agent.check_barren_plateau(0.1, threshold=1e-6)
        assert is_plateau is False
        assert msg is None
    
    def test_gradient_norms_tracked(self):
        """Test that gradient norms are tracked during training."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01)
        
        assert hasattr(agent, 'grad_norms')
        assert isinstance(agent.grad_norms, list)
        
        # After update, should have gradient norm
        env = create_env()
        state, _ = env.reset(seed=42)
        for _ in range(10):
            action = agent.select_action(normalize_state(state))
            state, reward, term, trunc, _ = env.step(action)
            agent.store_reward(reward)
            if term or trunc:
                break
        
        agent.update_policy()
        assert len(agent.grad_norms) > 0


class TestAdaptiveLearningRate:
    """Test adaptive learning rate scheduler."""
    
    def test_scheduler_created(self):
        """Test that scheduler is created when use_scheduler=True."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01, use_scheduler=True)
        assert agent.scheduler is not None
        assert agent.use_scheduler is True
    
    def test_scheduler_not_created(self):
        """Test that scheduler is None when use_scheduler=False."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01, use_scheduler=False)
        assert agent.scheduler is None
        assert agent.use_scheduler is False
    
    def test_learning_rate_reduction(self):
        """Test that learning rate reduces on plateau."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01, use_scheduler=True)
        
        initial_lr = agent.optimizer.param_groups[0]['lr']
        
        # Simulate plateau (same reward for > patience steps)
        for _ in range(60):  # patience=50
            agent.scheduler.step(50.0)
        
        new_lr = agent.optimizer.param_groups[0]['lr']
        
        assert new_lr < initial_lr
        assert abs(new_lr - initial_lr * 0.5) < 1e-6  # factor=0.5


class TestTrainingContinuation:
    """Test checkpoint save/load functionality."""
    
    def test_save_checkpoint_function_exists(self):
        """Test that save_checkpoint function exists in train.py."""
        from scripts.train import save_checkpoint
        assert callable(save_checkpoint)
    
    def test_load_checkpoint_function_exists(self):
        """Test that load_checkpoint function exists in train.py."""
        from scripts.train import load_checkpoint
        assert callable(load_checkpoint)
    
    def test_checkpoint_save_load_cycle(self):
        """Test saving and loading a checkpoint."""
        from scripts.train import save_checkpoint, load_checkpoint
        
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01)
        
        # Create some episode data
        episode = 100
        rewards = [10.0, 20.0, 30.0]
        
        # Save checkpoint to temp file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            save_checkpoint(agent, episode, rewards, agent.optimizer, temp_path)
            
            # Verify file exists
            assert Path(temp_path).exists()
            
            # Load checkpoint
            new_model = TinyMLP()
            checkpoint = load_checkpoint(temp_path, new_model)
            
            # Verify checkpoint contents
            assert checkpoint['episode'] == episode
            assert checkpoint['rewards'] == rewards  # Changed from episode_rewards
            assert 'model_state' in checkpoint
            assert 'optimizer_state' in checkpoint
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)


class TestVarianceMetrics:
    """Test variance and stability metrics."""
    
    def test_compute_variance_metrics_returns_dict(self):
        """Test that compute_variance_metrics returns a dictionary."""
        test_data = np.array([10, 20, 30, 40, 50])
        metrics = compute_variance_metrics(test_data)
        
        assert isinstance(metrics, dict)
    
    def test_variance_metrics_keys(self):
        """Test that all expected keys are present."""
        test_data = np.array([10, 20, 30, 40, 50])
        metrics = compute_variance_metrics(test_data)
        
        expected_keys = ['variance', 'std', 'mean', 'coefficient_of_variation',
                        'range', 'median', 'q25', 'q75']
        for key in expected_keys:
            assert key in metrics
    
    def test_variance_metrics_values(self):
        """Test that metrics are calculated correctly."""
        test_data = np.array([10, 20, 30, 40, 50])
        metrics = compute_variance_metrics(test_data)
        
        assert metrics['mean'] == 30.0
        assert metrics['median'] == 30.0
        assert metrics['range'] == 40.0
        assert metrics['q25'] == 20.0
        assert metrics['q75'] == 40.0
    
    def test_compute_stability_score(self):
        """Test stability score computation."""
        test_data = np.random.randn(200) + 50
        stability = compute_stability_score(test_data, window=100)
        
        assert isinstance(stability, (int, float))
        assert stability >= 0  # Stability should be non-negative


class TestStatisticalTests:
    """Test statistical test functions."""
    
    def test_compute_statistical_tests_exists(self):
        """Test that compute_statistical_tests function exists."""
        from scripts.benchmark import compute_statistical_tests
        assert callable(compute_statistical_tests)
    
    def test_statistical_tests_returns_dict(self):
        """Test that statistical tests return expected structure."""
        from scripts.benchmark import compute_statistical_tests
        
        # Create mock results with matching seeds
        classical = {1: list(range(500)), 2: list(range(500))}
        quantum = {1: list(range(500)), 2: list(range(500))}
        
        result = compute_statistical_tests(classical, quantum)
        
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'cohens_d' in result
        assert 't_statistic' in result
    
    def test_save_comparison_table_exists(self):
        """Test that save_comparison_table function exists."""
        from scripts.benchmark import save_comparison_table
        assert callable(save_comparison_table)


class TestAPIChanges:
    """Test that API changes are correctly implemented."""
    
    def test_update_policy_returns_tuple(self):
        """Test that update_policy returns (loss, grad_norm)."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01)
        
        # Generate episode data
        env = create_env()
        state, _ = env.reset(seed=42)
        for _ in range(10):
            action = agent.select_action(normalize_state(state))
            state, reward, term, trunc, _ = env.step(action)
            agent.store_reward(reward)
            if term or trunc:
                break
        
        result = agent.update_policy()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        loss, grad_norm = result
        assert isinstance(loss, (int, float))
        assert isinstance(grad_norm, (int, float))
    
    def test_train_returns_list(self):
        """Test that train returns list of rewards."""
        model = TinyMLP()
        agent = REINFORCEAgent(model, lr=0.01)
        env = create_env()
        
        result = agent.train(env=env, episodes=5, seed=42)
        
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(r, (int, float)) for r in result)


class TestOptionalScripts:
    """Test that optional scripts exist and are functional."""
    
    def test_validate_script_exists(self):
        """Test that validate.py exists."""
        validate_path = Path('scripts/validate.py')
        assert validate_path.exists()
    
    def test_hyperparameter_search_exists(self):
        """Test that hyperparameter_search.py exists."""
        hyperparam_path = Path('scripts/hyperparameter_search.py')
        assert hyperparam_path.exists()
    
    def test_validate_script_importable(self):
        """Test that validate.py can be imported."""
        try:
            import scripts.validate
            assert True
        except ImportError:
            pytest.fail("validate.py cannot be imported")
    
    def test_hyperparameter_script_importable(self):
        """Test that hyperparameter_search.py can be imported."""
        try:
            import scripts.hyperparameter_search
            assert True
        except ImportError:
            pytest.fail("hyperparameter_search.py cannot be imported")
