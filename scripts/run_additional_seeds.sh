#!/bin/bash
# Additional quantum training seeds for statistical robustness
# Seeds: 3 (retry), 5, 13, 17, 29

echo "Starting additional quantum training runs..."
echo "Seeds: 3 (retry), 5, 13, 17, 29"
echo "This will use 5 CPU cores on your i9-11980HK"
echo "Expected total time: ~15-20 minutes"
echo ""

cd /home/shazzy/projects/qml_project

echo "Starting seed 3 (retry)..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 3 --episodes 500 --diff_method backprop > logs/quantum_seed3_retry.log 2>&1 &
PID3=$!

echo "Starting seed 5..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 5 --episodes 500 --diff_method backprop > logs/quantum_seed5.log 2>&1 &
PID5=$!

echo "Starting seed 13..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 13 --episodes 500 --diff_method backprop > logs/quantum_seed13.log 2>&1 &
PID13=$!

echo "Starting seed 17..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 17 --episodes 500 --diff_method backprop > logs/quantum_seed17.log 2>&1 &
PID17=$!

echo "Starting seed 29..."
conda run -n qml-cartpole python -m scripts.train --mode quantum --seed 29 --episodes 500 --diff_method backprop > logs/quantum_seed29.log 2>&1 &
PID29=$!

echo ""
echo "All 5 training runs started!"
echo "Process IDs: $PID3 $PID5 $PID13 $PID17 $PID29"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/quantum_seed3_retry.log"
echo "  tail -f logs/quantum_seed5.log"
echo "  tail -f logs/quantum_seed13.log"
echo "  tail -f logs/quantum_seed17.log"
echo "  tail -f logs/quantum_seed29.log"
echo ""
echo "Wait for all processes to complete..."
wait $PID3 $PID5 $PID13 $PID17 $PID29

echo ""
echo "✓ All additional training runs complete!"
echo "Results saved to results/quantum/"

# Summary
echo ""
echo "=== Training Summary ==="
for seed in 3 5 13 17 29; do
    if [ -f "logs/quantum_seed${seed}.log" ] || [ -f "logs/quantum_seed${seed}_retry.log" ]; then
        logfile="logs/quantum_seed${seed}.log"
        [ -f "logs/quantum_seed${seed}_retry.log" ] && logfile="logs/quantum_seed${seed}_retry.log"
        
        solved=$(grep "Solved at episode" "$logfile" | tail -1 || echo "Not solved")
        final_avg=$(grep "Final 100-episode average" "$logfile" | tail -1 || echo "N/A")
        
        echo "Seed $seed: $solved | $final_avg"
    fi
done
