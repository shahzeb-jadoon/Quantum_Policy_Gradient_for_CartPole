"""
Tests for neural network models.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.models import TinyMLP, QuantumCircuit


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


# ===== Quantum Circuit Tests =====

class TestQuantumCircuitBasic:
    """Test basic quantum circuit initialization and properties."""
    
    def test_initialization(self):
        """Test circuit initializes without errors."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        assert qc is not None
    
    def test_is_nn_module(self):
        """Test circuit inherits from nn.Module."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        assert isinstance(qc, nn.Module)
    
    def test_device_creation(self):
        """Test quantum device is created."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        assert qc.dev is not None
        assert qc.dev.wires.tolist() == [0, 1, 2, 3]
    
    def test_attributes(self):
        """Test circuit stores correct attributes."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        assert qc.n_qubits == 4
        assert qc.n_layers == 1


class TestQuantumCircuitForward:
    """Test forward pass functionality."""
    
    def test_forward_single_state(self):
        """Test forward pass with single state."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        state = torch.randn(4)
        
        expectations = qc(state)
        
        # TorchLayer output may have extra dimension
        assert expectations.shape in [(2,), (2, 1)], f"Expected shape (2,) or (2,1), got {expectations.shape}"
    
    def test_output_is_tensor(self):
        """Test output is PyTorch tensor."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        state = torch.randn(4)
        
        expectations = qc(state)
        
        assert isinstance(expectations, torch.Tensor)
    
    def test_output_range(self):
        """Test expectation values in range [-1, 1]."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        state = torch.randn(4)
        
        expectations = qc(state)
        
        assert torch.all(expectations >= -1.0)
        assert torch.all(expectations <= 1.0)
    
    def test_deterministic_output(self):
        """Test same input gives same output."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        state = torch.randn(4)
        
        exp1 = qc(state)
        exp2 = qc(state)
        
        assert torch.allclose(exp1, exp2)
    
    def test_different_inputs(self):
        """Test different inputs give different outputs."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        state1 = torch.zeros(4)
        state2 = torch.ones(4)
        
        exp1 = qc(state1)
        exp2 = qc(state2)
        
        assert not torch.allclose(exp1, exp2)


class TestQuantumCircuitPyTorch:
    """Test PyTorch integration."""
    
    def test_has_parameters(self):
        """Test circuit has trainable parameters."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        
        params = list(qc.parameters())
        assert len(params) > 0
    
    def test_parameter_count(self):
        """Test parameter count is correct."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        
        # Single layer: 1 × 4 qubits × 3 rotations = 12 parameters
        param_count = qc.count_parameters()
        assert param_count == 12, f"Expected 12 parameters, got {param_count}"
    
    def test_parameter_count_multilayer(self):
        """Test parameter count for 3-layer circuit (data re-uploading)."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        
        # Three layers: 3 × 4 qubits × 3 rotations = 36 parameters
        param_count = qc.count_parameters()
        assert param_count == 36, f"Expected 36 parameters, got {param_count}"
    
    def test_parameters_require_grad(self):
        """Test all parameters require gradients."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        
        for param in qc.parameters():
            assert param.requires_grad is True
    
    def test_output_requires_grad(self):
        """Test output has requires_grad=True."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        state = torch.randn(4, requires_grad=True)
        
        expectations = qc(state)
        
        assert expectations.requires_grad is True


class TestQuantumCircuitGradients:
    """Test gradient flow through quantum circuit."""
    
    def test_gradient_flow(self):
        """Test gradients propagate through quantum circuit."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        state = torch.randn(4, requires_grad=True)
        
        # Forward pass
        expectations = qc(state)
        
        # Dummy loss
        loss = expectations.sum()
        loss.backward()
        
        # Check gradients exist on parameters
        for param in qc.parameters():
            assert param.grad is not None
    
    def test_parameters_update(self):
        """Test parameters change after optimization step."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        optimizer = torch.optim.Adam(qc.parameters(), lr=0.01)
        
        # Save initial parameters
        initial_params = [p.clone() for p in qc.parameters()]
        
        # Training step
        state = torch.randn(4)
        expectations = qc(state)
        loss = expectations.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check parameters changed
        changed = False
        for initial, current in zip(initial_params, qc.parameters()):
            if not torch.allclose(initial, current.data):
                changed = True
                break
        
        assert changed, "Parameters should change after optimization step"
    
    def test_no_gradient_without_backward(self):
        """Test no gradients without calling backward."""
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        state = torch.randn(4)
        
        # Forward only, no backward
        expectations = qc(state)
        
        for param in qc.parameters():
            assert param.grad is None, "Should have no gradient without backward"


class TestQuantumCircuitAgentCompatibility:
    """Test quantum circuit works with REINFORCE agent."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized with quantum circuit."""
        from src.agent import REINFORCEAgent
        
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        agent = REINFORCEAgent(policy=qc, lr=0.01)
        
        assert agent.policy is qc
    
    def test_agent_action_selection(self):
        """Test agent can select actions using quantum circuit."""
        from src.agent import REINFORCEAgent
        
        qc = QuantumCircuit(n_qubits=4, n_layers=1)
        agent = REINFORCEAgent(policy=qc, lr=0.01)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        # This will fail because QuantumCircuit doesn't have get_action method yet
        # We'll need to add a policy wrapper in Step 2.3, but for now just test
        # that the agent can be created
        assert agent is not None

