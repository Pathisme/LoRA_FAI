import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.training.baselines import prepare_baseline


def count_parameters(model):
    """Calculate total and trainable parameters."""

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_percentage": round(
            (trainable / total) * 100, 2
        ) if total else 0
    }


def load_config(config_path):
    """Load baseline configuration from YAML."""

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(model_name, num_labels):
    """Load a pretrained classification model."""

    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        raise ImportError(
            "Transformers is required. Install it using: "
            "pip install transformers"
        )

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )


def run_baselines(config_path, model_name=None):
    """Configure and compare the three baseline strategies."""

    config = load_config(config_path)

    model_config = config.get("model", {})
    selected_model = model_name or model_config.get("name")

    if not selected_model or selected_model == "your-base-model":
        raise ValueError(
            "Provide a real model name using --model."
        )

    num_labels = model_config.get("num_labels", 2)

    baseline_config = config.get("baselines", {})

    strategies = [
        "zero_shot",
        "linear_probe",
        "full_finetuning"
    ]

    results = {}

    print("\n========== BASELINE COMPARISON ==========\n")
    print(f"Model: {selected_model}")

    for strategy in strategies:

        settings = baseline_config.get(strategy, {})

        if not settings.get("enabled", False):
            print(f"\nSkipping {strategy}: disabled")
            continue

        print(f"\nPreparing: {strategy.upper()}")

        # Create a fresh model for each baseline.
        model = load_model(selected_model, num_labels)

        # Configure trainable parameters.
        prepare_baseline(model, strategy)

        # Collect parameter statistics.
        stats = count_parameters(model)

        results[strategy] = stats

        print(f"Total parameters: {stats['total_parameters']:,}")
        print(
            f"Trainable parameters: "
            f"{stats['trainable_parameters']:,}"
        )
        print(
            f"Frozen parameters: "
            f"{stats['frozen_parameters']:,}"
        )
        print(
            f"Trainable percentage: "
            f"{stats['trainable_percentage']}%"
        )

        del model

    # Display comparison.
    print("\n========== FINAL REPORT ==========\n")

    print(
        f"{'Strategy':<20}"
        f"{'Trainable Params':<20}"
        f"{'Trainable %':<15}"
    )

    for strategy, stats in results.items():
        print(
            f"{strategy:<20}"
            f"{stats['trainable_parameters']:<20,}"
            f"{stats['trainable_percentage']}%"
        )

    # Save results as JSON.
    output_dir = Path(
        config.get("output", {}).get(
            "results_directory",
            "outputs/baselines"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "baseline_report.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run baseline configuration comparisons."
    )

    parser.add_argument(
        "--config",
        default="configs/baselines.yaml",
        help="Path to baseline YAML configuration."
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Pretrained Hugging Face model name."
    )

    args = parser.parse_args()

    run_baselines(args.config, args.model)