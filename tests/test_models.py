"""Tests for the segmentation model architectures (src/ore_classifier/models.py)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.models import ResidualBlock, ResUNet, create_resunet  # noqa: E402


class ResidualBlockTest(unittest.TestCase):
    def test_preserves_spatial_size_and_maps_channels(self) -> None:
        block = ResidualBlock(3, 8)
        out = block(torch.randn(2, 3, 16, 16))
        self.assertEqual(tuple(out.shape), (2, 8, 16, 16))

    def test_identity_skip_when_channels_match(self) -> None:
        block = ResidualBlock(8, 8)
        out = block(torch.randn(1, 8, 12, 12))
        self.assertEqual(tuple(out.shape), (1, 8, 12, 12))


class ResUNetTest(unittest.TestCase):
    def test_forward_returns_two_class_logits_at_input_resolution(self) -> None:
        model = create_resunet(base_channels=4)
        self.assertIsInstance(model, ResUNet)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 2, 64, 64))

    def test_forward_handles_non_square_input(self) -> None:
        model = create_resunet(base_channels=4)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 32, 64))
        self.assertEqual(tuple(logits.shape), (1, 2, 32, 64))

    def test_gradients_flow_through_all_parameters(self) -> None:
        model = create_resunet(base_channels=4)
        logits = model(torch.randn(1, 3, 32, 32))
        loss = logits.mean()
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing, [])

    def test_base_channels_scales_parameter_count(self) -> None:
        small = sum(p.numel() for p in create_resunet(base_channels=4).parameters())
        large = sum(p.numel() for p in create_resunet(base_channels=8).parameters())
        self.assertGreater(large, small)


if __name__ == "__main__":
    unittest.main()
