# Quantum Policy Gradient for CartPole

**A rigorous, parameter-constrained comparison of Quantum vs. Classical Reinforcement Learning agents**

## Overview

This project implements a Quantum Policy Gradient (QPG) agent using a Variational Quantum Circuit (VQC) to solve the CartPole-v1 environment. The core scientific contribution is a fair comparison against a classical "Tiny MLP" with **matched parameter counts (~50 parameters)** to isolate whether quantum expressivity offers advantages in sample efficiency.

**Key Question**: Do quantum circuits offer superior expressivity per parameter compared to classical neural networks?

## Project Details

- **Environment**: CartPole-v1 (4D state space, 2 discrete actions)
- **Algorithm**: REINFORCE (Monte Carlo Policy Gradient)
- **Quantum Agent**: 4-qubit VQC with Data Re-uploading (~48 parameters)
- **Classical Baseline**: Tiny MLP (4→7→2 architecture, ~51 parameters)
- **Frameworks**: PennyLane, PyTorch, Gymnasium

## Installation

### Prerequisites
- Conda or Miniconda installed
- Linux/WSL (Ubuntu 24.04 recommended)

### Setup Environment

```bash
# Clone or navigate to project directory
cd ../qml_project

# Create conda environment from specification
conda env create -f environment.yml

# Activate environment
conda activate qml-cartpole

# Verify installation
pytest tests/test_setup.py -v
```

Expected output: All tests pass, confirming PyTorch, PennyLane, and Gymnasium are properly installed.

## Usage

### Train Classical Agent
```bash
python scripts/train.py --mode classical --seed 42 --episodes 500
```

### Train Quantum Agent (Fast Prototyping)
```bash
# Use backpropagation for fast gradient computation (simulator-only)
python scripts/train.py --mode quantum --seed 42 --episodes 500 --diff_method backprop
```

### Train Quantum Agent (Production with Parameter-Shift)
```bash
# Use parameter-shift rule for hardware-compatible gradients (slow but rigorous)
python scripts/train.py --mode quantum --seed 42 --episodes 500 --diff_method parameter-shift
```

### Generate Comparison Analysis
```bash
python scripts/benchmark.py --classical_dir results/classical --quantum_dir results/quantum
```

## Project Structure

```
qml_project/
├── src/                    # Source code modules
│   ├── env_wrapper.py     # CartPole environment utilities
│   ├── models.py          # TinyMLP and QuantumCircuit classes
│   ├── agent.py           # REINFORCE agent implementation
│   └── utils.py           # Logging, plotting, metrics
├── scripts/               # Executable training scripts
│   ├── train.py          # Main training entry point
│   └── benchmark.py      # Comparative analysis
├── tests/                 # Pytest test suite
├── results/               # Training outputs (git-ignored)
├── checkpoints/           # Model weights (git-ignored)
└── config.py             # Hyperparameters
```

## Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_models.py -v
pytest tests/test_agent.py -v
```

## Hardware Requirements

- **Minimum**: 4-core CPU, 8GB RAM
- **Recommended**: 8-core CPU, 16GB RAM (for parallel runs)
- **Optimized for**: Intel i9-11980HK with AVX512 support (2-3x speedup for quantum simulation)

## Key Features

- **Fair Parameter Comparison**: Strict 45-55 parameter budget for both agents
- **Data Re-uploading**: Non-linear quantum feature maps via interleaved encoding
- **Two Measurement Strategies**: Softmax (primary) and Parity (fallback)
- **Barren Plateau Monitoring**: Automatic gradient norm tracking
- **Statistical Rigor**: 5-seed experiments with confidence intervals

## 🏆 Results Summary

**Status:** Complete & Validated

| Metric | Classical MLP | Quantum VQC | Result |
|--------|---------------|-------------|--------|
| **Success Rate** | 100% (9/9 seeds) | 89% (8/9 seeds) | **Parity** (High Reliability) |
| **Convergence** | 267 ± 87 eps | 275 ± 59 eps | **No Significant Diff** (p=0.81) |
| **Parameters** | 51 | 48 | **6% Reduction** (Efficiency) |
| **Hardware Valid?** | N/A | **Yes** (Parameter-Shift) | **Physically Realizable** |

**Key Conclusion:** The Quantum Agent matches Classical performance with **6% fewer parameters**, validated via rigorous paired t-tests and hardware-compatible parameter-shift gradients.

### Demonstration Videos

- **Classical Agent:** 30-second continuous balancing ([quantum_agent_demo.gif](report_assets/quantum_agent_demo.gif))
- **Quantum Agent:** 30-second continuous balancing ([classical_agent_demo.gif](report_assets/classical_agent_demo.gif))

Both agents successfully balance the pole for 1500 consecutive timesteps with HUD overlay showing step counts.

### Generated Artifacts

All results, plots, and trained models available in:
- `results/` - Training curves, metrics, comparative analysis
- `checkpoints/` - Trained model weights
- `report_assets/` - Publication-ready figures and demos
- `REPORT.md` - Complete analysis and findings (coming soon)

## References

1. Jerbi et al. (2021) - "Parametrized Quantum Policies for Reinforcement Learning" (NeurIPS)
2. Pérez-Salinas et al. (2020) - "Data re-uploading for a universal quantum classifier" (Quantum)
3. McClean et al. (2018) - "Barren plateaus in quantum neural network training landscapes" (Nature Comm.)

## License

Academic project for CSCI-739 Quantum Machine Learning course.

## Author

Shahzeb Jadoon
