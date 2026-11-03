"""
Baseline preparation utilities for LoRA/QLoRA experiments.

Supported baselines:
1. Zero-Shot
2. Linear Probe / Head-Only Fine-Tuning
3. Full Fine-Tuning

This module configures which model parameters are trainable.
It does not implement the training loop.
"""

import torch.nn as nn


BASELINE_TYPES = {
    "zero_shot",
    "linear_probe",
    "full_finetuning"
}


def count_model_parameters(model):
    """Calculate total, trainable and frozen parameters."""

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = total_parameters - trainable_parameters

    percentage = (
        trainable_parameters / total_parameters * 100
        if total_parameters > 0 else 0
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "trainable_percentage": round(percentage, 4)
    }


def find_classification_head(model):
    """
    Automatically identify a common classification head.

    Users can provide an explicit module name if their
    model uses a different architecture.
    """

    candidates = [
        "classifier",
        "classification_head",
        "score",
        "head"
    ]

    module_names = [
        name for name, _ in model.named_modules()
        if name
    ]

    for candidate in candidates:
        matches = [
            name for name in module_names
            if name == candidate or name.endswith("." + candidate)
        ]

        if matches:
            # Select the deepest matching module.
            return max(matches, key=len)

    raise ValueError(
        "Classification head not found. "
        "Provide its name using head_module_name."
    )


def configure_zero_shot(model):
    """
    Configure the model for zero-shot inference.

    All parameters are frozen.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return count_model_parameters(model)


def configure_linear_probe(model, head_module_name=None):
    """
    Configure head-only training.

    The pretrained backbone remains frozen.
    Only the classification head is trainable.
    """

    if head_module_name is None:
        head_module_name = find_classification_head(model)

    modules = dict(model.named_modules())

    if head_module_name not in modules:
        raise ValueError(
            f"Module '{head_module_name}' does not exist in the model."
        )

    # Freeze the complete model.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze only the selected classification head.
    head_module = modules[head_module_name]

    for parameter in head_module.parameters():
        parameter.requires_grad = True

    model.train()

    return count_model_parameters(model)


def configure_full_finetuning(model):
    """
    Configure full fine-tuning.

    All model parameters become trainable.
    """

    for parameter in model.parameters():
        parameter.requires_grad = True

    model.train()

    return count_model_parameters(model)


def prepare_baseline(
    model,
    strategy,
    head_module_name=None
):
    """
    Configure a model using the selected baseline strategy.

    Args:
        model: PyTorch model.
        strategy: zero_shot, linear_probe or full_finetuning.
        head_module_name: Optional classification head name.

    Returns:
        Dictionary containing strategy and parameter statistics.
    """

    strategy = strategy.lower().strip()

    if strategy not in BASELINE_TYPES:
        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            f"Choose from {BASELINE_TYPES}."
        )

    if strategy == "zero_shot":
        statistics = configure_zero_shot(model)

    elif strategy == "linear_probe":
        statistics = configure_linear_probe(
            model,
            head_module_name
        )

    else:
        statistics = configure_full_finetuning(model)

    return {
        "baseline": strategy,
        **statistics
    }


def compare_baselines(model_factory):
    """
    Generate parameter-count comparisons for all baselines.

    model_factory must return a NEW model each time.
    This prevents one baseline's parameter settings from
    affecting another baseline.
    """

    results = {}

    for strategy in BASELINE_TYPES:
        model = model_factory()

        results[strategy] = prepare_baseline(
            model,
            strategy
        )

    return results


def print_baseline_report(results):
    """Print a formatted baseline comparison."""

    print("\n========== BASELINE COMPARISON ==========")

    for name, stats in results.items():

        print(f"\nBaseline: {name.upper()}")

        print(
            f"Total Parameters: "
            f"{stats['total_parameters']:,}"
        )

        print(
            f"Trainable Parameters: "
            f"{stats['trainable_parameters']:,}"
        )

        print(
            f"Frozen Parameters: "
            f"{stats['frozen_parameters']:,}"
        )

        print(
            f"Trainable Percentage: "
            f"{stats['trainable_percentage']}%"
        )

    print("\n=========================================")