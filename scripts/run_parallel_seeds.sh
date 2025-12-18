#!/bin/bash
# Parallel quantum training for seeds 1-4
# Run this script to train all 4 seeds simultaneously on your i9's 8 cores
# Each terminal will show its own progress with periodic checkpoints

echo "Starting parallel quantum training for seeds 1-4..."
echo "This will use 4 CPU cores (~50% of your i9-11980HK)"
echo "Expected total time: ~15-20 minutes"
echo ""

# Create tmux session or use gnome-terminal tabs
# Option 1: Using background processes (simplest)
cd /home/shazzy/projects/qml_project

echo "Starting seed 1..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 1 --episodes 500 --diff_method backprop > logs/quantum_seed1.log 2>&1 &
PID1=$!

echo "Starting seed 2..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 2 --episodes 500 --diff_method backprop > logs/quantum_seed2.log 2>&1 &
PID2=$!

echo "Starting seed 3..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 3 --episodes 500 --diff_method backprop > logs/quantum_seed3.log 2>&1 &
PID3=$!

echo "Starting seed 4..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 4 --episodes 500 --diff_method backprop > logs/quantum_seed4.log 2>&1 &
PID4=$!

echo ""
echo "All 4 training runs started!"
echo "Process IDs: $PID1 $PID2 $PID3 $PID4"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/quantum_seed1.log"
echo "  tail -f logs/quantum_seed2.log"
echo "  tail -f logs/quantum_seed3.log"
echo "  tail -f logs/quantum_seed4.log"
echo ""
echo "Or check results files:"
echo "  tail -f results/quantum/seed1_rewards.json"
echo ""
echo "Wait for all processes to complete..."
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "✓ All training runs complete!"
echo "Results saved to results/quantum/"
