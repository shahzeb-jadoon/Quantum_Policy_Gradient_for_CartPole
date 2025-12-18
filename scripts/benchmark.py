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
        if len(rolling) < max_len:
            rolling = np.pad(rolling, (0, max_len - len(rolling)), mode='edge')
        quantum_curves.append(rolling)
    
    quantum_curves = np.array(quantum_curves)
    quantum_mean = np.mean(quantum_curves, axis=0)
    quantum_std = np.std(quantum_curves, axis=0, ddof=1)  # Sample std
    
    episodes = np.arange(1, max_len + 1)
    
    # Plot quantum with confidence interval
    plt.plot(episodes, quantum_mean, 'r-', linewidth=2, 
             label=f'Quantum (n={len(quantum_results)})')
    plt.fill_between(episodes, 
                     quantum_mean - quantum_std,
                     quantum_mean + quantum_std,
                     color='red', alpha=0.2)
    
    # Process and plot classical
    if classical_results:
        if len(classical_results) == 1:
            # Single seed - just plot the line
            for seed, rewards in classical_results.items():
                rolling = compute_rolling_mean(rewards)
                plt.plot(rolling, 'b-', linewidth=2, 
                        label=f'Classical (seed {seed})')
        else:
            # Multiple seeds - show mean and confidence interval
            classical_curves = []
            for rewards in classical_results.values():
                rolling = compute_rolling_mean(rewards)
                if len(rolling) < max_len:
                    rolling = np.pad(rolling, (0, max_len - len(rolling)), mode='edge')
                classical_curves.append(rolling)
            
            classical_curves = np.array(classical_curves)
            classical_mean = np.mean(classical_curves, axis=0)
            classical_std = np.std(classical_curves, axis=0, ddof=1)  # Sample std
            
            plt.plot(episodes, classical_mean, 'b-', linewidth=2, 
                    label=f'Classical (n={len(classical_results)})')
            plt.fill_between(episodes,
                           classical_mean - classical_std,
                           classical_mean + classical_std,
                           color='blue', alpha=0.2)
    
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
    """Print statistical summary with all seeds and solved-only breakdown."""
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*70)
    
    # Classical
    print("\n📘 CLASSICAL BASELINE (TinyMLP, 51 parameters)")
    print("-" * 70)
    
    classical_solve_eps = []
    classical_final_avgs = []
    classical_solved = 0
    
    for seed in sorted(classical_results.keys()):
        rewards = classical_results[seed]
        solve_ep = None
        for i in range(99, len(rewards)):
            if np.mean(rewards[i-99:i+1]) >= 195:
                solve_ep = i + 1
                break
        final_avg = np.mean(rewards[-100:])
        classical_final_avgs.append(final_avg)
        
        if solve_ep:
            classical_solve_eps.append(solve_ep)
            classical_solved += 1
            status = f"Solved at {solve_ep}"
        else:
            status = "Did NOT solve"
        print(f"Seed {seed:2d}: {status}, Final avg: {final_avg:.1f}")
    
    # Quantum
    print("\n📕 QUANTUM (VQC + Data Re-uploading, 42 parameters)")
    print("-" * 70)
    
    quantum_solve_eps = []
    quantum_final_avgs = []
    quantum_solved = 0
    
    for seed in sorted(quantum_results.keys()):
        rewards = quantum_results[seed]
        solve_ep = None
        for i in range(99, len(rewards)):
            if np.mean(rewards[i-99:i+1]) >= 195:
                solve_ep = i + 1
                break
        
        final_avg = np.mean(rewards[-100:])
        quantum_final_avgs.append(final_avg)
        
        if solve_ep:
            quantum_solve_eps.append(solve_ep)
            quantum_solved += 1
            print(f"Seed {seed:2d}: Solved at {solve_ep:3d}, Final avg: {final_avg:6.1f}")
        else:
            print(f"Seed {seed:2d}: Did NOT solve,   Final avg: {final_avg:6.1f}")
    
    # Statistics
    print("\n📊 STATISTICAL COMPARISON")
    print("-" * 70)
    
    # Success rates
    c_rate = 100 * classical_solved / len(classical_results)
    q_rate = 100 * quantum_solved / len(quantum_results)
    print(f"Classical success rate: {classical_solved}/{len(classical_results)} ({c_rate:.0f}%)")
    print(f"Quantum success rate:   {quantum_solved}/{len(quantum_results)} ({q_rate:.0f}%)")
    
    # Episodes to solve (solved runs only)
    if classical_solve_eps and quantum_solve_eps:
        c_mean = np.mean(classical_solve_eps)
        c_std = np.std(classical_solve_eps, ddof=1)  # Sample std
        q_mean = np.mean(quantum_solve_eps)
        q_std = np.std(quantum_solve_eps, ddof=1)  # Sample std
        
        print(f"\n📈 Episodes to Solve (Solved Runs Only):")
        print(f"Classical: {c_mean:.1f} ± {c_std:.1f} (n={len(classical_solve_eps)})")
        print(f"Quantum:   {q_mean:.1f} ± {q_std:.1f} (n={len(quantum_solve_eps)})")
        
        diff = q_mean - c_mean
        pct = (diff / c_mean) * 100
        print(f"Difference: {diff:+.1f} episodes ({pct:+.1f}%)")
    
    # Final performance (solved runs only)
    c_solved_avgs = [classical_final_avgs[i] for i in range(len(classical_final_avgs)) 
                     if i < len(classical_solve_eps)]
    q_solved_avgs = [quantum_final_avgs[i] for i, seed in enumerate(sorted(quantum_results.keys())) 
                     if any(j == i for j in range(len(quantum_solve_eps)))]
    
    # Actually get solved avgs correctly
    c_solved_final = []
    for seed in sorted(classical_results.keys()):
        rewards = classical_results[seed]
        solve_ep = None
        for i in range(99, len(rewards)):
            if np.mean(rewards[i-99:i+1]) >= 195:
                solve_ep = i + 1
                break
        if solve_ep:
            c_solved_final.append(np.mean(rewards[-100:]))
    
    q_solved_final = []
    for seed in sorted(quantum_results.keys()):
        rewards = quantum_results[seed]
        solve_ep = None
        for i in range(99, len(rewards)):
            if np.mean(rewards[i-99:i+1]) >= 195:
                solve_ep = i + 1
                break
        if solve_ep:
            q_solved_final.append(np.mean(rewards[-100:]))
    
    if c_solved_final and q_solved_final:
        c_perf_mean = np.mean(c_solved_final)
        c_perf_std = np.std(c_solved_final, ddof=1)
        q_perf_mean = np.mean(q_solved_final)
        q_perf_std = np.std(q_solved_final, ddof=1)
        
        print(f"\n📈 Final Performance (Solved Runs Only, 100-ep avg):")
        print(f"Classical: {c_perf_mean:.1f} ± {c_perf_std:.1f}")
        print(f"Quantum:   {q_perf_mean:.1f} ± {q_perf_std:.1f}")
    
    # All seeds stats
    print(f"\n📊 All Seeds Performance (Including Failed):")
    c_all_mean = np.mean(classical_final_avgs)
    c_all_std = np.std(classical_final_avgs, ddof=1)
    q_all_mean = np.mean(quantum_final_avgs)
    q_all_std = np.std(quantum_final_avgs, ddof=1)
    print(f"Classical: {c_all_mean:.1f} ± {c_all_std:.1f}")
    print(f"Quantum:   {q_all_mean:.1f} ± {q_all_std:.1f}")
    
    print(f"\n📋 Parameter Count:")
    print(f"Classical: 51 parameters")
    print(f"Quantum:   42 parameters (18% fewer)")
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
