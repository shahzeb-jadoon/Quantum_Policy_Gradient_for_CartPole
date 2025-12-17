"""
Setup verification tests.

This module tests that all required dependencies are properly installed
and that the CartPole environment is accessible.

Run with: pytest tests/test_setup.py -v
"""

import torch
import pennylane as qml
import gymnasium as gym
import matplotlib
import pytest


def test_torch_import():
    """Verify PyTorch is installed and accessible."""
    assert torch.__version__ is not None
    print(f"✓ PyTorch version: {torch.__version__}")


def test_pennylane_import():
    """Verify PennyLane is installed and accessible."""
    assert qml.__version__ is not None
    print(f"✓ PennyLane version: {qml.__version__}")


def test_gymnasium_import():
    """Verify Gymnasium is installed and accessible."""
    assert gym.__version__ is not None
    print(f"✓ Gymnasium version: {gym.__version__}")


def test_matplotlib_import():
    """Verify Matplotlib is installed for plotting."""
    assert matplotlib.__version__ is not None
    print(f"✓ Matplotlib version: {matplotlib.__version__}")


def test_cartpole_env():
    """Verify CartPole-v1 environment can be created and reset."""
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    
    # CartPole state should be 4-dimensional
    assert len(obs) == 4, f"Expected state dimension 4, got {len(obs)}"
    
    # State should contain: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    assert obs.shape == (4,), f"Expected shape (4,), got {obs.shape}"
    
    print(f"✓ CartPole-v1 initialized successfully with state: {obs}")
    env.close()


def test_cartpole_step():
    """Verify CartPole environment can execute a step."""
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    
    # Take a random action (0 = push left, 1 = push right)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert len(next_obs) == 4, "Next state should be 4-dimensional"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(terminated, bool), "Terminated flag should be boolean"
    assert isinstance(truncated, bool), "Truncated flag should be boolean"
    
    print(f"✓ CartPole step executed: action={action}, reward={reward}")
    env.close()


def test_pennylane_device():
    """Verify PennyLane quantum device can be created."""
    dev = qml.device('default.qubit', wires=4)
    # PennyLane 0.42+ uses dev.wires instead of dev.num_wires
    num_wires = len(dev.wires) if hasattr(dev, 'wires') else dev.num_wires
    assert num_wires == 4, f"Expected 4 qubits, got {num_wires}"
    print(f"✓ PennyLane device created with {num_wires} qubits")


def test_torch_tensor_operations():
    """Verify PyTorch tensor operations work correctly."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    # Test basic operations
    z = x + y
    assert z.shape == (4,)
    
    # Test gradient computation
    x_grad = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = (x_grad ** 2).sum()
    loss.backward()
    assert x_grad.grad is not None
    
    print(f"✓ PyTorch tensor operations and autograd working")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
