#!/usr/bin/env python3
"""
Extract best hyperparameter configuration from logs.
"""

import re

print('='*80)
print('BEST HYPERPARAMETER CONFIGURATION FROM PARTIAL RESULTS')
print('='*80)

with open('logs/hyperparam_search.log') as f:
    content = f.read()

# Extract configuration from header
print('\nSearch Parameters:')
print('-'*80)
lr_match = re.search(r'Learning rates: \[([\d., ]+)\]', content)
gamma_match = re.search(r'Gamma values: \[([\d., ]+)\]', content)
clip_match = re.search(r'Grad clip values: \[([\d., ]+)\]', content)
depth_match = re.search(r'Circuit depths: \[([\d., ]+)\]', content)

if lr_match:
    print(f'Learning rates: [{lr_match.group(1)}]')
if gamma_match:
    print(f'Gamma values: [{gamma_match.group(1)}]')
if clip_match:
    print(f'Grad clip values: [{clip_match.group(1)}]')
if depth_match:
    print(f'Circuit depths: [{depth_match.group(1)}]')

# Get best from grid search tracker
grid_lines = re.findall(r'Grid search:\s+(\d+)%.*?(\d+)/(\d+).*?best_mean=([\d.]+), best_lr=([\d.]+)', content)

if grid_lines:
    last = grid_lines[-1]
    percent, current, total, best_mean, best_lr = last
    
    print(f'\n{current}/{total} configurations completed ({percent}%)')
    print()
    print('BEST CONFIGURATION FOUND:')
    print('-'*80)
    print(f'Mean Reward: {best_mean}')
    print(f'Learning Rate: {best_lr}')
    print()
    print('Note: Full configuration details (gamma, grad_clip, depth) require')
    print('      parsing individual run results from the logs or waiting for')
    print('      the partial_results.json file to be created.')
    print()
    print('Recommendation: Use lr=0.005 based on current best, with default')
    print('                values for other parameters until more data available')
else:
    print('No grid search progress found in logs')

print('='*80)
