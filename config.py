"""
Centralized hyperparameters and configuration for Quantum Policy Gradient CartPole project.

This module contains all training hyperparameters, architecture settings, and experimental
configurations. Modify values here rather than hardcoding them in training scripts.
"""

# Random seeds for reproducibility (5 seeds for statistical significance)
SEEDS = [1, 2, 3, 4, 5]

# Training hyperparameters
MAX_EPISODES = 500  # Total episodes to train
GAMMA = 0.99  # Discount factor for return calculation
LR_CLASSICAL = 0.01  # Learning rate for classical MLP
LR_QUANTUM = 0.01  # Learning rate for quantum VQC (might need tuning)
CHECKPOINT_FREQ = 50  # Save model checkpoint every N episodes

# Classical MLP architecture
CLASSICAL_HIDDEN_SIZE = 7  # Hidden layer neurons (4→7→2 = 51 params)

# Quantum VQC architecture
QUANTUM_LAYERS = 3  # Number of data re-uploading layers
QUANTUM_SHOTS = 1024  # Number of measurement shots per forward pass
QUANTUM_N_QUBITS = 4  # Number of qubits (matches CartPole state dimension)

# CartPole environment
ENV_NAME = "CartPole-v1"
SOLVED_THRESHOLD = 195.0  # Average reward over 100 episodes to consider "solved"

# Logging and output
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
LOG_INTERVAL = 10  # Print training stats every N episodes
