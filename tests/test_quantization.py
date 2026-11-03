import unittest
import torch
import torch.nn as nn

from src.core.qlora_layer import QLoRALinear


class TestQLoRAQuantization(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

        self.base_layer = nn.Linear(16, 8)

        self.qlora_layer = QLoRALinear(
            base_layer=self.base_layer,
            rank=4,
            alpha=8,
            dropout=0.0
        )

        self.input_data = torch.randn(5, 16)

    def test_base_layer_is_frozen(self):
        """Base model parameters should not be trainable."""

        for param in self.qlora_layer.base_layer.parameters():
            self.assertFalse(param.requires_grad)

    def test_adapter_parameters_are_trainable(self):
        """LoRA adapter parameters should be trainable."""

        self.assertTrue(self.qlora_layer.lora_A.requires_grad)
        self.assertTrue(self.qlora_layer.lora_B.requires_grad)

    def test_forward_output_shape(self):
        """Output dimensions should match the base layer."""

        output = self.qlora_layer(self.input_data)

        self.assertEqual(output.shape, (5, 8))

    def test_initial_output_matches_base_layer(self):
        """Initially, zero-initialized B should make adapter output zero."""

        with torch.no_grad():
            expected = self.base_layer(self.input_data)
            actual = self.qlora_layer(self.input_data)

        torch.testing.assert_close(actual, expected)

    def test_adapter_receives_gradients(self):
        """Adapter parameters should receive gradients during training."""

        output = self.qlora_layer(self.input_data)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(self.qlora_layer.lora_A.grad)
        self.assertIsNotNone(self.qlora_layer.lora_B.grad)

        self.assertFalse(
            any(
                param.grad is not None
                for param in self.qlora_layer.base_layer.parameters()
            )
        )

    def test_trainable_parameter_count(self):
        """Trainable parameters should equal rank * (input + output)."""

        expected = 4 * (16 + 8)
        actual = self.qlora_layer.trainable_parameters()

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()