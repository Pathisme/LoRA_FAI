import torch
import torch.nn as nn

from src.core.qlora_layer import QLoRALinear
from src.training.checkpoint import (
    save_adapter_checkpoint,
    load_adapter_checkpoint,
    print_checkpoint_report
)


# Create a simple model with a QLoRA layer.
class DemoModel(nn.Module):

    def __init__(self):
        super().__init__()

        base_layer = nn.Linear(100, 50)

        self.layer = QLoRALinear(
            base_layer=base_layer,
            rank=8,
            alpha=16
        )

    def forward(self, x):
        return self.layer(x)


# Original model.
model = DemoModel()

# Save adapter.
checkpoint_info = save_adapter_checkpoint(
    model,
    "adapter_checkpoint",
    adapter_config={
        "rank": 8,
        "alpha": 16,
        "quantization": "NF4"
    }
)

print_checkpoint_report(checkpoint_info)


# Create another model with the same architecture.
new_model = DemoModel()

# Load saved adapter.
load_info = load_adapter_checkpoint(
    new_model,
    "adapter_checkpoint"
)

print("\nLoading Result:")
print(load_info)


# Verify restored adapter weights.
original = model.layer.lora_A.detach()
restored = new_model.layer.lora_A.detach()

assert torch.equal(original, restored)

print("\nAdapter checkpoint test passed!")