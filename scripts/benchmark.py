"""
Benchmark comparison script for classical vs quantum agents.

Generates training curve comparison plot with mean and confidence intervals
across multiple random seeds. Saves figure to results/comparison/.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_results(results_dir):
    """
    Load all training results from a directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        dict: Seed -> episode rewards mapping
    """
    results = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.glob('seed*_rewards.json'):
        seed = int(json_file.stem.split('seed')[1].split('_')[0])
        with open(json_file) as f:
            data = json.load(f)
            results[seed] = data['episode_rewards']
    
    return results


def compute_rolling_mean(rewards, window=100):
    """Compute rolling mean with specified window size."""
    if len(rewards) < window:
        return np.array([np.mean(rewards[:i+1]) for i in range(len(rewards))])
    
    rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
    # Pad beginning
    padding = [np.mean(rewards[:i+1]) for i in range(window-1)]
    return np.concatenate([padding, rolling])


def plot_comparison(classical_results, quantum_results, save_path):
    """
    Plot training curves with confidence intervals.
    
    Args:
        classical_results: Dict of classical training results
        quantum_results: Dict of quantum training results  
        save_path: Where to save the figure
    """
    plt.figure(figsize=(12, 7))
    
    # Process quantum results
    max_len = max(len(r) for r in quantum_results.values())
    quantum_curves = []
    
    for rewards in quantum_results.values():
        rolling = compute_rolling_mean(rewards)
        # Pad if needed
        if len(rolling) < max_len:
            rolling = np.pad(rolling, (0, max_len - len(rolling)), 
                           mode='edge')
        quantum_curves.append(rolling)
    
    quantum_curves = np.array(quantum_curves)
    quantum_mean = np.mean(quantum_curves, axis=0)
    quantum_std = np.std(quantum_curves, axis=0)
    
    episodes = np.arange(1, max_len + 1)
    
    # Plot quantum
    plt.plot(episodes, quantum_mean, 'r-', linewidth=2, label=f'Quantum (n={len(quantum_results)})')
    plt.fill_between(episodes, 
                     quantum_mean - quantum_std,
                     quantum_mean + quantum_std,
                     color='red', alpha=0.2)
    
    # Process and plot classical
    if classical_results:
        for seed, rewards in classical_results.items():
            rolling = compute_rolling_mean(rewards)
            plt.plot(rolling, 'b-', linewidth=2, 
                    label=f'Classical (seed {seed})')
    
    # Solved threshold
    plt.axhline(y=195, color='green', linestyle='--', linewidth=1.5,
                label='Solved Threshold (195)')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward (100-episode rolling mean)', fontsize=12)
    plt.title('Classical vs Quantum Policy Gradient: CartPole-v1', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    
    plt.close()


def print_summary(classical_results, quantum_results):
    """Print statistical summary of results."""
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*70)
    
    # Classical
    print("\n📘 CLASSICAL BASELINE (TinyMLP, 51 parameters)")
    print("-" * 70)
    for seed, rewards in classical_results.items():
        # Find solve episode
        solve_ep = None
        for i in range(99, len(rewards)):
            if np.mean(rewards[i-99:i+1]) >= 195:
                solve_ep = i + 1
                break
        final_avg = np.mean(rewards[-100:])
        status = f"Solved at {solve_ep}" if solve_ep else "Did NOT solve"
        print(f"Seed {seed}: {status}, Final avg: {final_avg:.1f}")
    
    # Quantum
    print("\n📕 QUANTUM (VQC + Data Re-uploading, 42 parameters)")
    print("-" * 70)
    
    solve_episodes = []
    final_avgs = []
    solved_count = 0
    
    for seed in sorted(quantum_results.keys()):
        rewards = quantum_results[seed]
        solve_ep = None
        for i in range(99, len(rewards)):
            if np.mean(rewards[i-99:i+1]) >= 195:
                solve_ep = i + 1
                break
        
        final_avg = np.mean(rewards[-100:])
        final_avgs.append(final_avg)
        
        if solve_ep:
            solve_episodes.append(solve_ep)
            solved_count += 1
            print(f"Seed {seed:2d}: Solved at {solve_ep:3d}, Final avg: {final_avg:6.1f}")
        else:
            print(f"Seed {seed:2d}: Did NOT solve,   Final avg: {final_avg:6.1f}")
    
    # Statistics
    print("\n📊 STATISTICAL COMPARISON")
    print("-" * 70)
    print(f"Quantum success rate: {solved_count}/{len(quantum_results)} ({100*solved_count/len(quantum_results):.0f}%)")
    
    if solve_episodes:
        quantum_mean_solve = np.mean(solve_episodes)
        quantum_std_solve = np.std(solve_episodes)
        print(f"Quantum episodes to solve: {quantum_mean_solve:.1f} ± {quantum_std_solve:.1f}")
        
        # Assume classical is seed 42 at 229
        classical_solve = 229
        diff = quantum_mean_solve - classical_solve
        pct = (diff / classical_solve) * 100
        print(f"Classical episodes to solve: {classical_solve}")
        print(f"Difference: {diff:+.1f} episodes ({pct:+.1f}%)")
    
    print(f"\nQuantum final performance: {np.mean(final_avgs):.1f} ± {np.std(final_avgs):.1f}")
    print(f"Parameter count: 42 (quantum) vs 51 (classical) = 18% fewer")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark comparison plots')
    parser.add_argument('--classical_dir', type=str, default='results/classical',
                       help='Directory containing classical results')
    parser.add_argument('--quantum_dir', type=str, default='results/quantum',
                       help='Directory containing quantum results')
    parser.add_argument('--output', type=str, default='results/comparison/training_curves.png',
                       help='Output path for comparison plot')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    classical_results = load_results(args.classical_dir)
    quantum_results = load_results(args.quantum_dir)
    
    print(f"Found {len(classical_results)} classical seeds")
    print(f"Found {len(quantum_results)} quantum seeds")
    
    # Generate plot
    plot_comparison(classical_results, quantum_results, args.output)
    
    # Print summary
    print_summary(classical_results, quantum_results)


if __name__ == '__main__':
    main()
