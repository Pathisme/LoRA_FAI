"""
QLoRA Linear Layer.

Features:
- Frozen 4-bit quantized base layer.
- NF4 quantization support.
- Double quantization support.
- Trainable LoRA A and B matrices.
- Configurable LoRA rank, alpha and dropout.

The base layer is quantized using bitsandbytes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLoRALinear(nn.Module):
    """
    QLoRA implementation wrapping a quantized linear layer.

    Formula:

        Output = W_quantized(x) + (alpha / rank) * B(A(x))

    The quantized base weights remain frozen.
    Only LoRA A and B matrices are trainable.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("Rank must be greater than zero.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be between 0 and 1.")

        if not hasattr(base_layer, "in_features"):
            raise TypeError("Base layer must have in_features.")

        if not hasattr(base_layer, "out_features"):
            raise TypeError("Base layer must have out_features.")

        self.base_layer = base_layer

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze the quantized base model parameters.
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        # Trainable LoRA matrices.
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features)
        )

        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )

        # Initialize A using Kaiming initialization.
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=5 ** 0.5
        )

        # Dropout applied to adapter input.
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized base and LoRA adapter.
        """

        # Original quantized model output.
        base_output = self.base_layer(x)

        # Adapter computation.
        adapter_input = self.lora_dropout(x)

        adapter_input = adapter_input.to(self.lora_A.dtype)

        adapter_output = F.linear(
            adapter_input,
            self.lora_A
        )

        adapter_output = F.linear(
            adapter_output,
            self.lora_B
        )

        # Match the base output dtype.
        adapter_output = adapter_output.to(base_output.dtype)

        # Combine base model and adapter outputs.
        return base_output + self.scaling * adapter_output

    def trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def total_parameters(self) -> int:
        """Return total parameters in this layer."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling}"
        )


def quantize_linear_layer(
    linear_layer: nn.Linear,
    compute_dtype=None,
    quant_type: str = "nf4",
    double_quant: bool = True,
):
    """
    Convert a standard PyTorch Linear layer into a
    bitsandbytes 4-bit Linear4bit layer.

    NF4 and double quantization are enabled by default.

    Quantization is finalized when the layer is moved
    to its target device, typically CUDA.
    """

    try:
        import bitsandbytes as bnb

    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for 4-bit quantization. "
            "Install a compatible version before using this function."
        ) from exc

    if not isinstance(linear_layer, nn.Linear):
        raise TypeError("Expected a torch.nn.Linear layer.")

    if quant_type not in ("nf4", "fp4"):
        raise ValueError("quant_type must be 'nf4' or 'fp4'.")

    if compute_dtype is None:
        compute_dtype = torch.float16

    # Create a bitsandbytes 4-bit linear layer.
    quantized_layer = bnb.nn.Linear4bit(
        input_features=linear_layer.in_features,
        output_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=double_quant,
        quant_type=quant_type,
    )

    # Transfer the original full-precision weights.
    quantized_layer.load_state_dict(linear_layer.state_dict())

    # Move to target device to finalize quantization.
    # On CUDA, bitsandbytes performs 4-bit quantization.
    target_device = linear_layer.weight.device
    quantized_layer = quantized_layer.to(target_device)

    return quantized_layer
