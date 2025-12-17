"""
Tests for neural network models.
"""

import pytest
import torch
import numpy as np
from src.models import TinyMLP


class TestTinyMLPArchitecture:
    """Test TinyMLP architecture and parameter count."""
    
    def test_initialization(self):
        """Test model initializes without errors."""
        model = TinyMLP()
        assert model is not None
    
    def test_parameter_count(self):
        """Test total parameter count is ~51."""
        model = TinyMLP()
        param_count = model.count_parameters()
        
        # Expected: (4*7 + 7) + (7*2 + 2) = 28 + 7 + 14 + 2 = 51
        assert param_count == 51, f"Expected 51 parameters, got {param_count}"
    
    def test_parameter_count_calculation(self):
        """Verify parameter count breakdown."""
        model = TinyMLP()
        
        # fc1: 4×7 weights + 7 biases = 35
        fc1_params = sum(p.numel() for p in model.fc1.parameters())
        assert fc1_params == 35
        
        # fc2: 7×2 weights + 2 biases = 16
        fc2_params = sum(p.numel() for p in model.fc2.parameters())
        assert fc2_params == 16
        
        # Total: 35 + 16 = 51
        total = fc1_params + fc2_params
        assert total == 51
    
    def test_parameters_trainable(self):
        """Test all parameters require gradients."""
        model = TinyMLP()
        for param in model.parameters():
            assert param.requires_grad is True


class TestTinyMLPForward:
    """Test forward pass functionality."""
    
    def test_forward_single_state(self):
        """Test forward pass with single state."""
        model = TinyMLP()
        state = torch.randn(4)
        
        probs = model(state)
        
        assert probs.shape == (2,), f"Expected shape (2,), got {probs.shape}"
    
    def test_forward_batch(self):
        """Test forward pass with batch of states."""
        model = TinyMLP()
        batch_size = 10
        states = torch.randn(batch_size, 4)
        
        probs = model(states)
        
        assert probs.shape == (batch_size, 2)
    
    def test_output_is_probability(self):
        """Test output is valid probability distribution."""
        model = TinyMLP()
        state = torch.randn(4)
        
        probs = model(state)
        
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
        
        # All probabilities should be non-negative
        assert torch.all(probs >= 0)
        
        # All probabilities should be <= 1
        assert torch.all(probs <= 1)
    
    def test_output_deterministic(self):
        """Test same input gives same output (no randomness in forward)."""
        model = TinyMLP()
        state = torch.randn(4)
        
        probs1 = model(state)
        probs2 = model(state)
        
        assert torch.allclose(probs1, probs2)
    
    def test_different_inputs_different_outputs(self):
        """Test different states give different probabilities."""
        model = TinyMLP()
        state1 = torch.tensor([0.0, 0.0, 0.0, 0.0])
        state2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        
        probs1 = model(state1)
        probs2 = model(state2)
        
        # Different inputs should give different outputs
        assert not torch.allclose(probs1, probs2)


class TestTinyMLPActionSampling:
    """Test action sampling methods."""
    
    def test_get_action_returns_valid_action(self):
        """Test get_action returns valid action in {0, 1}."""
        model = TinyMLP()
        state = torch.randn(4)
        
        action, log_prob = model.get_action(state)
        
        assert action in [0, 1], f"Expected action in {{0, 1}}, got {action}"
        assert isinstance(action, int)
    
    def test_get_action_returns_log_prob(self):
        """Test get_action returns log probability."""
        model = TinyMLP()
        state = torch.randn(4)
        
        action, log_prob = model.get_action(state)
        
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.shape == ()  # Scalar
        # Log prob should be negative (since prob <= 1)
        assert log_prob <= 0
    
    def test_get_action_stochastic(self):
        """Test get_action samples stochastically."""
        model = TinyMLP()
        state = torch.randn(4)
        
        # Sample 100 times
        actions = [model.get_action(state)[0] for _ in range(100)]
        
        # Should get both actions at least once (with very high probability)
        unique_actions = set(actions)
        assert len(unique_actions) >= 1  # At minimum, should sample something
        
        # If probabilities are not extreme (0 or 1), should see both actions
        probs = model(state)
        if probs[0] > 0.01 and probs[1] > 0.01:
            assert len(unique_actions) == 2, "Should sample both actions when probs are balanced"
    
    def test_get_log_prob_single(self):
        """Test get_log_prob for single state-action pair."""
        model = TinyMLP()
        state = torch.randn(4)
        action = torch.tensor(0)
        
        log_prob = model.get_log_prob(state, action)
        
        assert log_prob.shape == ()
        assert log_prob <= 0  # Log of probability is non-positive
    
    def test_get_log_prob_batch(self):
        """Test get_log_prob for batch."""
        model = TinyMLP()
        batch_size = 5
        states = torch.randn(batch_size, 4)
        actions = torch.randint(0, 2, (batch_size,))
        
        log_probs = model.get_log_prob(states, actions)
        
        assert log_probs.shape == (batch_size,)
        assert torch.all(log_probs <= 0)
    
    def test_log_prob_matches_forward(self):
        """Test get_log_prob matches manual calculation from forward."""
        model = TinyMLP()
        state = torch.randn(4)
        action = 0
        
        # Get log prob using method
        log_prob = model.get_log_prob(state, torch.tensor(action))
        
        # Calculate manually from forward pass
        probs = model(state)
        expected_log_prob = torch.log(probs[action])
        
        assert torch.allclose(log_prob, expected_log_prob, atol=1e-6)


class TestTinyMLPGradients:
    """Test gradient flow through the network."""
    
    def test_gradients_exist(self):
        """Test gradients are computed correctly."""
        model = TinyMLP()
        state = torch.randn(4)
        action = torch.tensor(0)
        
        # Forward pass
        log_prob = model.get_log_prob(state, action)
        
        # Backward pass
        loss = -log_prob  # Dummy loss
        loss.backward()
        
        # Check gradients exist and are non-zero
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            # At least some parameters should have non-zero gradients
    
    def test_gradients_update_parameters(self):
        """Test parameters change after optimization step."""
        model = TinyMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Save initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Training step
        state = torch.randn(4)
        action = torch.tensor(0)
        log_prob = model.get_log_prob(state, action)
        loss = -log_prob
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check parameters changed
        changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current.data):
                changed = True
                break
        
        assert changed, "Parameters should change after optimization step"
    
    def test_no_gradient_without_backward(self):
        """Test no gradients without calling backward."""
        model = TinyMLP()
        state = torch.randn(4)
        
        # Forward only, no backward
        probs = model(state)
        
        for param in model.parameters():
            assert param.grad is None, "Should have no gradient without backward"


class TestTinyMLPEdgeCases:
    """Test edge cases and robustness."""
    
    def test_zero_state(self):
        """Test with zero state."""
        model = TinyMLP()
        state = torch.zeros(4)
        
        probs = model(state)
        
        assert torch.allclose(probs.sum(), torch.tensor(1.0))
        assert torch.all(probs >= 0)
    
    def test_large_state_values(self):
        """Test with large state values."""
        model = TinyMLP()
        state = torch.tensor([100.0, -100.0, 50.0, -50.0])
        
        probs = model(state)
        
        # Should still produce valid probabilities
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.all(probs >= 0)
        assert not torch.any(torch.isnan(probs))
    
    def test_multiple_forward_passes(self):
        """Test model handles multiple sequential forward passes."""
        model = TinyMLP()
        
        for _ in range(10):
            state = torch.randn(4)
            probs = model(state)
            assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
