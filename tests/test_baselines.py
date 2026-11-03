import torch.nn as nn

from src.training.baselines import (
    prepare_baseline,
    print_baseline_report
)


class DemoModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.classifier = nn.Linear(32, 5)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# Test Zero-Shot
zero_shot_model = DemoModel()

zero_shot = prepare_baseline(
    zero_shot_model,
    "zero_shot"
)

# Test Linear Probe
linear_model = DemoModel()

linear_probe = prepare_baseline(
    linear_model,
    "linear_probe"
)

# Test Full Fine-Tuning
full_model = DemoModel()

full_finetuning = prepare_baseline(
    full_model,
    "full_finetuning"
)


results = {
    "Zero-Shot": zero_shot,
    "Linear Probe": linear_probe,
    "Full Fine-Tuning": full_finetuning
}

print_baseline_report(results)


# Verify expected behaviour.
assert zero_shot["trainable_parameters"] == 0

assert (
    linear_probe["trainable_parameters"]
    < full_finetuning["trainable_parameters"]
)

assert (
    full_finetuning["trainable_parameters"]
    == full_finetuning["total_parameters"]
)

print("\nAll baseline tests passed!")