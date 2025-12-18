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
from scipy import stats
from src.utils import compute_variance_metrics


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


def compute_statistical_tests(classical_results, quantum_results):
    """
    Perform automated statistical tests (paired t-test).
    
    Args:
        classical_results: Dict of classical training results
        quantum_results: Dict of quantum training results
        
    Returns:
        dict: Statistical test results
    """
    # Get final 100-episode averages for each seed
    classical_final = []
    quantum_final = []
    
    # Match seeds
    common_seeds = sorted(set(classical_results.keys()) & set(quantum_results.keys()))
    
    for seed in common_seeds:
        c_rewards = classical_results[seed]
        q_rewards = quantum_results[seed]
        classical_final.append(np.mean(c_rewards[-100:]))
        quantum_final.append(np.mean(q_rewards[-100:]))
    
    if len(common_seeds) < 2:
        return {
            'test_type': 'paired_t_test',
            'n_pairs': len(common_seeds),
            'error': 'Insufficient paired samples for t-test (need >= 2)'
        }
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(quantum_final, classical_final)
    
    # Effect size (Cohen's d for paired samples)
    differences = np.array(quantum_final) - np.array(classical_final)
    diff_std = np.std(differences, ddof=1)
    
    # Handle edge case: if std is 0, all differences are identical
    if diff_std == 0:
        # If mean is also 0, there's no difference (d = 0)
        # If mean is non-zero, effect is infinite (use large value)
        cohens_d = 0.0 if np.mean(differences) == 0 else np.inf
    else:
        cohens_d = np.mean(differences) / diff_std
    
    results = {
        'test_type': 'paired_t_test',
        'n_pairs': len(common_seeds),
        'common_seeds': common_seeds,
        'classical_mean': np.mean(classical_final),
        'classical_std': np.std(classical_final, ddof=1),
        'quantum_mean': np.mean(quantum_final),
        'quantum_std': np.std(quantum_final, ddof=1),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_difference': np.mean(differences),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01
    }
    
    return results


def save_comparison_table(classical_results, quantum_results, stats_test, save_path):
    """
    Save comparison metrics table as markdown file.
    
    Args:
        classical_results: Dict of classical results
        quantum_results: Dict of quantum results
        stats_test: Statistical test results from compute_statistical_tests()
        save_path: Path to save markdown file
    """
    # Compute variance metrics
    all_classical = [r for rewards in classical_results.values() for r in rewards]
    all_quantum = [r for rewards in quantum_results.values() for r in rewards]
    
    c_metrics = compute_variance_metrics(all_classical)
    q_metrics = compute_variance_metrics(all_quantum)
    
    # Create markdown table
    md_content = []
    md_content.append("# Classical vs Quantum Comparison: CartPole-v1")
    md_content.append("")
    md_content.append("## Model Specifications")
    md_content.append("")
    md_content.append("| Model | Architecture | Parameters |")
    md_content.append("|-------|--------------|------------|")
    md_content.append("| Classical | TinyMLP (4→7→2) | 51 |")
    md_content.append("| Quantum | VQC + Data Re-uploading (3 layers) | 42 |")
    md_content.append("")
    md_content.append("## Performance Metrics")
    md_content.append("")
    md_content.append("| Metric | Classical | Quantum |")
    md_content.append("|--------|-----------|---------|")
    md_content.append(f"| Number of Seeds | {len(classical_results)} | {len(quantum_results)} |")
    md_content.append(f"| Mean Reward | {c_metrics['mean']:.2f} | {q_metrics['mean']:.2f} |")
    md_content.append(f"| Std Dev (sample) | {c_metrics['std']:.2f} | {q_metrics['std']:.2f} |")
    md_content.append(f"| Median | {c_metrics['median']:.2f} | {q_metrics['median']:.2f} |")
    md_content.append(f"| Min | {c_metrics['min']:.2f} | {q_metrics['min']:.2f} |")
    md_content.append(f"| Max | {c_metrics['max']:.2f} | {q_metrics['max']:.2f} |")
    md_content.append(f"| Q25 | {c_metrics['q25']:.2f} | {q_metrics['q25']:.2f} |")
    md_content.append(f"| Q75 | {c_metrics['q75']:.2f} | {q_metrics['q75']:.2f} |")
    md_content.append(f"| Coefficient of Variation | {c_metrics['coefficient_of_variation']:.4f} | {q_metrics['coefficient_of_variation']:.4f} |")
    md_content.append("")
    md_content.append("## Statistical Test Results")
    md_content.append("")
    
    if 'error' in stats_test:
        md_content.append(f"**Error:** {stats_test['error']}")
    else:
        md_content.append(f"**Test:** Paired t-test (n={stats_test['n_pairs']} matched seeds)")
        md_content.append("")
        md_content.append("| Statistic | Value |")
        md_content.append("|-----------|-------|")
        md_content.append(f"| t-statistic | {stats_test['t_statistic']:.4f} |")
        md_content.append(f"| p-value | {stats_test['p_value']:.4f} |")
        md_content.append(f"| Cohen's d | {stats_test['cohens_d']:.4f} |")
        md_content.append(f"| Mean Difference (Q - C) | {stats_test['mean_difference']:.2f} |")
        md_content.append(f"| Significant (α=0.05) | {'✓ Yes' if stats_test['significant_05'] else '✗ No'} |")
        md_content.append(f"| Significant (α=0.01) | {'✓ Yes' if stats_test['significant_01'] else '✗ No'} |")
        md_content.append("")
        md_content.append("### Interpretation")
        md_content.append("")
        if stats_test['significant_05']:
            direction = "better" if stats_test['mean_difference'] > 0 else "worse"
            md_content.append(f"The quantum agent performs **statistically significantly {direction}** ")
            md_content.append(f"than the classical agent (p={stats_test['p_value']:.4f} < 0.05).")
        else:
            md_content.append("No statistically significant difference found between quantum and classical agents ")
            md_content.append(f"(p={stats_test['p_value']:.4f} ≥ 0.05).")
        md_content.append("")
        effect_interpretation = (
            "negligible" if abs(stats_test['cohens_d']) < 0.2 else
            "small" if abs(stats_test['cohens_d']) < 0.5 else
            "medium" if abs(stats_test['cohens_d']) < 0.8 else
            "large"
        )
        md_content.append(f"Effect size (Cohen's d = {stats_test['cohens_d']:.2f}): **{effect_interpretation}**")
    
    md_content.append("")
    md_content.append(f"*Generated automatically by benchmark.py*")
    
    # Write to file
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(md_content))
    
    print(f"Comparison table saved to {save_path}")


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
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    stats_test = compute_statistical_tests(classical_results, quantum_results)
    
    # Print statistical test results
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)
    if 'error' in stats_test:
        print(f"Error: {stats_test['error']}")
    else:
        print(f"Test: Paired t-test (n={stats_test['n_pairs']} matched seeds)")
        print(f"t-statistic: {stats_test['t_statistic']:.4f}")
        print(f"p-value: {stats_test['p_value']:.4f}")
        print(f"Cohen's d: {stats_test['cohens_d']:.4f}")
        print(f"Mean difference (Q - C): {stats_test['mean_difference']:.2f}")
        print(f"Significant at α=0.05: {'Yes ✓' if stats_test['significant_05'] else 'No ✗'}")
        print(f"Significant at α=0.01: {'Yes ✓' if stats_test['significant_01'] else 'No ✗'}")
    print("="*70)
    
    # Save comparison table
    table_path = Path(args.output).parent / 'comparison_metrics.md'
    save_comparison_table(classical_results, quantum_results, stats_test, table_path)
    
    # Print summary
    print_summary(classical_results, quantum_results)


if __name__ == '__main__':
    main()
