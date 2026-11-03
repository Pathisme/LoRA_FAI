"""
Adapter Checkpoint Management for LoRA and QLoRA.

Features:
1. Save only LoRA adapter parameters.
2. Load adapter parameters into an existing model.
3. Save adapter configuration as JSON.
4. Calculate checkpoint file size.
5. Support LoRA and QLoRA layers using lora_A and lora_B.
"""

import json
import os
from pathlib import Path

import torch


def extract_adapter_weights(model):
    """
    Extract only LoRA adapter weights from a model.

    Supports layers containing lora_A and lora_B.
    Base model weights are excluded.
    """

    adapter_state = {}

    for name, module in model.named_modules():

        for adapter_name in ("lora_A", "lora_B"):

            if hasattr(module, adapter_name):

                parameter = getattr(module, adapter_name)

                if isinstance(parameter, torch.Tensor):

                    key = (
                        f"{name}.{adapter_name}"
                        if name else adapter_name
                    )

                    adapter_state[key] = (
                        parameter.detach().cpu().clone()
                    )

    if not adapter_state:
        raise ValueError(
            "No LoRA adapter parameters found in the model."
        )

    return adapter_state


def save_adapter_checkpoint(
    model,
    save_directory,
    adapter_config=None
):
    """
    Save only adapter weights and configuration.

    Args:
        model: Model containing LoRA/QLoRA layers.
        save_directory: Directory for checkpoint files.
        adapter_config: Optional adapter configuration.

    Returns:
        Dictionary containing saved file information.
    """

    save_path = Path(save_directory)

    save_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # Extract adapter-only weights.
    adapter_state = extract_adapter_weights(model)

    checkpoint_file = save_path / "adapter_model.pt"

    config_file = save_path / "adapter_config.json"

    # Save adapter parameters only.
    torch.save(
        adapter_state,
        checkpoint_file
    )

    # Default configuration.
    config = adapter_config or {}

    config.setdefault("format", "LoRA/QLoRA adapter")
    config.setdefault("rank", None)
    config.setdefault("alpha", None)
    config.setdefault("quantization", None)

    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            indent=4
        )

    checkpoint_size = os.path.getsize(checkpoint_file)

    return {
        "checkpoint_path": str(checkpoint_file),
        "config_path": str(config_file),
        "adapter_tensors": len(adapter_state),
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_size_mb": round(
            checkpoint_size / (1024 ** 2), 4
        )
    }


def load_adapter_checkpoint(
    model,
    checkpoint_path,
    map_location="cpu"
):
    """
    Load adapter weights into an existing model.

    The pretrained base model must already be loaded
    and have compatible LoRA adapter layers.
    """

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        checkpoint_path = (
            checkpoint_path / "adapter_model.pt"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load adapter-only checkpoint.
    try:
        adapter_state = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True
        )

    except TypeError:
        # Compatibility with older PyTorch versions.
        adapter_state = torch.load(
            checkpoint_path,
            map_location=map_location
        )

    # Current model parameters.
    model_state = model.state_dict()

    # Validate adapter compatibility.
    missing_keys = [
        key for key in adapter_state
        if key not in model_state
    ]

    if missing_keys:
        raise ValueError(
            f"Incompatible adapter checkpoint. "
            f"Missing keys: {missing_keys}"
        )

    for key, tensor in adapter_state.items():

        if model_state[key].shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"model={model_state[key].shape}, "
                f"checkpoint={tensor.shape}"
            )

    # Update only adapter weights.
    model.load_state_dict(
        adapter_state,
        strict=False
    )

    return {
        "loaded_tensors": len(adapter_state),
        "checkpoint_path": str(checkpoint_path),
        "status": "success"
    }


def get_checkpoint_size(checkpoint_path):
    """Return checkpoint size in bytes and MB."""

    path = Path(checkpoint_path)

    if path.is_dir():
        path = path / "adapter_model.pt"

    if not path.exists():
        raise FileNotFoundError(path)

    size_bytes = path.stat().st_size

    return {
        "size_bytes": size_bytes,
        "size_mb": round(
            size_bytes / (1024 ** 2), 4
        )
    }


def print_checkpoint_report(checkpoint_info):
    """Display checkpoint information."""

    print("\n========== ADAPTER CHECKPOINT ==========")

    for key, value in checkpoint_info.items():
        print(f"{key}: {value}")

    print("========================================\n")