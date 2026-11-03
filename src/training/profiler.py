"""
Training profiler for LoRA and QLoRA experiments.

Tracks:
1. GPU memory usage (allocated and reserved).
2. Training execution time.
3. Training throughput (tokens per second).
4. Total and trainable model parameters.

Supports CPU-only environments and optional PyTorch installation.
"""

import time
from typing import Optional
import random


class TrainingTimer:
    """Measures the time taken by a training operation."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0

    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")

        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

        return self.elapsed_time

    def get_elapsed_time(self):
        """Return elapsed time in seconds."""
        return self.elapsed_time


class VRAMProfiler:
    """Monitors GPU memory usage using PyTorch."""

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.cuda_available = False
        self.started = False

        self.initial_allocated = 0
        self.initial_reserved = 0
        self.peak_allocated = 0
        self.peak_reserved = 0

        try:
            import torch

            self.torch = torch

            if torch.cuda.is_available():
                self.cuda_available = True
                self.device = device or "cuda"

        except ImportError:
            self.torch = None

    def start(self):
        """Start monitoring GPU memory."""

        if not self.cuda_available:
            print("CUDA unavailable. VRAM monitoring disabled.")
            return

        torch = self.torch

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        self.initial_allocated = torch.cuda.memory_allocated(
            self.device
        )

        self.initial_reserved = torch.cuda.memory_reserved(
            self.device
        )

        self.started = True

    def stop(self):
        """Capture final and peak GPU memory usage."""

        if not self.cuda_available or not self.started:
            return None

        torch = self.torch

        torch.cuda.synchronize(self.device)

        self.peak_allocated = torch.cuda.max_memory_allocated(
            self.device
        )

        self.peak_reserved = torch.cuda.max_memory_reserved(
            self.device
        )

        self.started = False

        return self.report()

    def report(self):
        """Return GPU memory statistics in MB."""

        if not self.cuda_available:
            return {
                "gpu_available": False,
                "message": "CUDA is not available."
            }

        mb = 1024 ** 2

        return {
            "gpu_available": True,
            "initial_allocated_mb": round(
                self.initial_allocated / mb, 2
            ),
            "initial_reserved_mb": round(
                self.initial_reserved / mb, 2
            ),
            "peak_allocated_mb": round(
                self.peak_allocated / mb, 2
            ),
            "peak_reserved_mb": round(
                self.peak_reserved / mb, 2
            )
        }


def calculate_throughput(total_tokens: int, elapsed_seconds: float):
    """
    Calculate training throughput in tokens per second.

    Args:
        total_tokens: Number of processed tokens.
        elapsed_seconds: Total training duration.

    Returns:
        Tokens processed per second.
    """

    if elapsed_seconds <= 0:
        raise ValueError("Elapsed time must be greater than zero.")

    if total_tokens < 0:
        raise ValueError("Total tokens cannot be negative.")

    return total_tokens / elapsed_seconds


def count_parameters(model):
    """
    Count total and trainable model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary containing parameter statistics.
    """

    total_params = 0
    trainable_params = 0

    for parameter in model.parameters():
        count = parameter.numel()

        total_params += count

        if parameter.requires_grad:
            trainable_params += count

    non_trainable_params = total_params - trainable_params

    trainable_percentage = (
        (trainable_params / total_params) * 100
        if total_params > 0 else 0
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "trainable_percentage": round(trainable_percentage, 4)
    }


def print_trainable_parameters(model):
    """Display model parameter statistics."""

    stats = count_parameters(model)

    print("\n========== MODEL PARAMETERS ==========")

    print(f"Total parameters:       {stats['total_parameters']:,}")

    print(
        f"Trainable parameters:   "
        f"{stats['trainable_parameters']:,}"
    )

    print(
        f"Non-trainable parameters: "
        f"{stats['non_trainable_parameters']:,}"
    )

    print(
        f"Trainable percentage:   "
        f"{stats['trainable_percentage']}%"
    )

    print("======================================\n")

    return stats

def simulate_vram_usage(
    model_name="QLoRA",
    min_memory=4.0,
    max_memory=6.0
):
    """
    Generate illustrative VRAM usage.

    NOTE: These are simulated values, not actual GPU measurements.
    """

    memory_used = round(
        random.uniform(min_memory, max_memory), 2
    )

    return {
        "model": model_name,
        "simulated_vram_gb": memory_used,
        "measurement_type": "SIMULATED"
    }

def simulate_training_metrics(
    total_tokens=1000000,
    model_name="QLoRA"
):
    """
    Generate illustrative training performance metrics.

    NOTE: These are simulated values, not actual training results.
    """

    training_time = round(random.uniform(400, 500), 2)

    throughput = round(
        total_tokens / training_time, 2
    )

    return {
        "model": model_name,
        "training_time_seconds": training_time,
        "total_tokens": total_tokens,
        "throughput_tokens_per_second": throughput,
        "measurement_type": "SIMULATED"
    }

def generate_comparison_report():
    """
    Generate an illustrative LoRA vs QLoRA comparison.
    """

    lora_vram = simulate_vram_usage(
        "LoRA", 7.0, 9.0
    )

    qlora_vram = simulate_vram_usage(
        "QLoRA", 4.0, 6.0
    )

    lora_metrics = simulate_training_metrics(
        model_name="LoRA"
    )

    qlora_metrics = simulate_training_metrics(
        model_name="QLoRA"
    )

    print("\n========== LoRA vs QLoRA COMPARISON ==========")

    print("\nMetric                 LoRA          QLoRA")

    print(
        f"VRAM (GB)              "
        f"{lora_vram['simulated_vram_gb']:<14}"
        f"{qlora_vram['simulated_vram_gb']}"
    )

    print(
        f"Training Time (sec)    "
        f"{lora_metrics['training_time_seconds']:<14}"
        f"{qlora_metrics['training_time_seconds']}"
    )

    print(
        f"Throughput (tokens/s)  "
        f"{lora_metrics['throughput_tokens_per_second']:<14}"
        f"{qlora_metrics['throughput_tokens_per_second']}"
    )

    print("\nNOTE: ALL VALUES ARE SIMULATED.")
    print("==============================================\n")

    return {
        "LoRA": {
            **lora_vram,
            **lora_metrics
        },
        "QLoRA": {
            **qlora_vram,
            **qlora_metrics
        }
    }