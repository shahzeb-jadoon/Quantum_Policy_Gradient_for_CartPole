#!/bin/bash
# Train classical baseline on all matching quantum seeds
# for rigorous paired statistical comparison

echo "Training classical baseline on matching quantum seeds..."
echo "Seeds: 1, 2, 3, 4, 5, 13, 17, 29"
echo "Expected time: ~15-20 minutes total"
echo ""

cd /home/shazzy/projects/qml_project

for seed in 1 2 3 4 5 13 17 29; do
    echo "=== Training Classical Seed $seed ==="
    conda run -n qml-cartpole python -m scripts.train --mode classical --seed $seed --episodes 500
    echo ""
done

echo "✓ All classical seeds complete!"
echo ""
echo "Classical results now available for seeds: 1-5, 13, 17, 29, 42"
echo "This enables rigorous paired comparison with quantum results."
