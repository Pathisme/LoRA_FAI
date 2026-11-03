from src.training.profiler import (
    TrainingTimer,
    VRAMProfiler,
    calculate_throughput
)

# Test 1: Training Timer
timer = TrainingTimer()

timer.start()

# Simulate training for 2 seconds
import time
time.sleep(2)

elapsed = timer.stop()

print("\nTraining Time:", round(elapsed, 2), "seconds")


# Test 2: Throughput
tokens = 10000

throughput = calculate_throughput(tokens, elapsed)

print("Throughput:", round(throughput, 2), "tokens/second")


# Test 3: GPU Memory
profiler = VRAMProfiler()

profiler.start()

print("\nGPU Memory Statistics:")

print(profiler.stop())

from src.training.profiler import generate_comparison_report

results = generate_comparison_report()

print("Generated comparison successfully!")