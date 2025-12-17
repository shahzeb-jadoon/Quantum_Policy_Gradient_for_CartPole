"""
Neural network models for CartPole policy learning.

This module contains policy network architectures:
- TinyMLP: Classical baseline (4→7→2, ~51 parameters)
- QuantumCircuit: Quantum VQC baseline (4-qubit, ~45-55 parameters)
- QuantumPolicy: Quantum policy with Softmax measurement (~42 parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


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


class QuantumCircuit(nn.Module):
    """
    4-qubit Variational Quantum Circuit for CartPole policy with data re-uploading.
    
    Uses PennyLane with PyTorch integration via qml.qnn.TorchLayer.
    
    Architecture (Data Re-uploading):
    - Interleave AngleEmbedding and StronglyEntanglingLayers
    - Structure: (Encode → Variational) × n_layers
    - Measurements: Expectation values ⟨Z₀⟩ and ⟨Z₁⟩
    
    Parameters: n_layers × n_qubits × 3 rotations per layer
    - Single layer (n_layers=1): 1 × 4 × 3 = 12 parameters
    - Three layers (n_layers=3): 3 × 4 × 3 = 36 parameters
    """
    
    def __init__(self, n_qubits=4, n_layers=1):
        """
        Initialize quantum circuit.
        
        Args:
            n_qubits (int): Number of qubits (default: 4 for CartPole)
            n_layers (int): Number of variational layers (default: 1)
        """
        super(QuantumCircuit, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Define weight shapes for TorchLayer
        # StronglyEntanglingLayers: (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Create QNode first, then wrap in TorchLayer
        qnode = qml.QNode(self._circuit, self.dev)
        
        # Wrap QNode in TorchLayer for PyTorch compatibility
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    
    def _circuit(self, inputs, weights):
        """
        Quantum circuit implementation with data re-uploading.
        
        Data re-uploading interleaves state encoding and variational layers,
        creating non-linear transformations and richer feature representations.
        
        Structure: (AngleEmbedding → StronglyEntanglingLayers) × n_layers
        
        Args:
            inputs (torch.Tensor): State variables, shape (4,)
            weights (torch.Tensor): Variational parameters, shape (n_layers, n_qubits, 3)
            
        Returns:
            tuple: (⟨Z₀⟩, ⟨Z₁⟩) expectation values
        """
        # Interleave encoding and variational layers
        for layer in range(self.n_layers):
            # Re-upload state data
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            
            # Apply variational transformation for this layer
            qml.StronglyEntanglingLayers(weights[layer:layer+1], wires=range(self.n_qubits))
        
        # Measure expectation values on first two qubits
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
    
    def forward(self, state):
        """
        Forward pass through quantum circuit.
        
        Args:
            state (torch.Tensor): CartPole state, shape (4,) or (batch, 4)
            
        Returns:
            torch.Tensor: Expectation values, shape (2,) or (batch, 2)
        """
        # TorchLayer handles conversion to tensor and gradient tracking
        expectations = self.qlayer(state)
        
        # Stack expectations into tensor if needed
        if isinstance(expectations, tuple):
            expectations = torch.stack(expectations, dim=-1)
        
        return expectations
    
    def count_parameters(self):
        """
        Count total trainable parameters.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QuantumPolicy(nn.Module):
    """
    Quantum policy network wrapping QuantumCircuit with hybrid output layer.
    
    Combines quantum circuit expectation values with classical linear layer
    and Softmax to produce action probabilities for REINFORCE.
    
    Architecture:
    - QuantumCircuit: Produces 2 expectation values ⟨Z₀⟩, ⟨Z₁⟩
    - Hybrid Linear Layer: Maps 2 expectations → 2 logits
    - Softmax: Converts logits to action probabilities
    
    Total parameters: quantum_params + hybrid_params
    - With L=3 layers: 36 (circuit) + 6 (hybrid: 2×2 weights + 2 biases) = 42 params
    
    This is the "Softmax measurement strategy" (primary approach).
    """
    
    def __init__(self, n_qubits=4, n_layers=3):
        """
        Initialize quantum policy with hybrid output.
        
        Args:
            n_qubits (int): Number of qubits (default: 4)
            n_layers (int): Number of data re-uploading layers (default: 3)
        """
        super(QuantumPolicy, self).__init__()
        
        # Quantum circuit component
        self.quantum_circuit = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        
        # Hybrid linear layer: 2 expectations → 2 action logits
        # Parameters: 2×2 weights + 2 biases = 6 parameters
        self.hybrid_layer = nn.Linear(2, 2)
    
    def forward(self, state):
        """
        Forward pass through quantum-classical hybrid network.
        
        Args:
            state (torch.Tensor): CartPole state, shape (4,) or (batch, 4)
            
        Returns:
            torch.Tensor: Action probabilities, shape (2,) or (batch, 2)
        """
        # Get expectation values from quantum circuit
        expectations = self.quantum_circuit(state)
        
        # Ensure expectations has correct shape for linear layer
        if expectations.dim() == 2 and expectations.shape[-1] == 1:
            expectations = expectations.squeeze(-1)
        
        # Map expectations to logits via hybrid layer
        logits = self.hybrid_layer(expectations)
        
        # Convert to probabilities via softmax
        probs = F.softmax(logits,dim=-1)
        
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
        Count total trainable parameters (quantum + hybrid).
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


