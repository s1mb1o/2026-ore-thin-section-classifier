from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.model_io import (
    _checkpoint_load_error,
    create_segformer_for_eval,
    forward_logits,
    load_binary_segmentation_checkpoint,
    remap_segformer_state_dict_for_model,
    resolve_device,
)
from ore_classifier.models import create_resunet


def tensor(*shape: int) -> torch.Tensor:
    return torch.zeros(shape)


class ModelIOTest(unittest.TestCase):
    def test_remaps_segformer_stages_checkpoint_to_encoder_model_namespace(self) -> None:
        checkpoint_state = {
            "segformer.stages.0.patch_embeddings.proj.weight": tensor(64, 3, 7, 7),
            "segformer.stages.0.patch_embeddings.layer_norm.weight": tensor(64),
            "segformer.stages.1.blocks.2.layernorm_before.bias": tensor(128),
            "segformer.stages.1.blocks.2.attention.q_proj.weight": tensor(128, 128),
            "segformer.stages.1.blocks.2.attention.sequence_reduction.sequence_reduction.weight": tensor(128, 128, 4, 4),
            "segformer.stages.1.blocks.2.attention.sequence_reduction.layer_norm.weight": tensor(128),
            "segformer.stages.1.blocks.2.attention.o_proj.bias": tensor(128),
            "segformer.stages.1.blocks.2.layernorm_after.weight": tensor(128),
            "segformer.stages.1.blocks.2.mlp.fc1.weight": tensor(512, 128),
            "segformer.stages.1.blocks.2.mlp.fc2.bias": tensor(128),
            "segformer.stages.3.layer_norm.bias": tensor(512),
            "decode_head.linear_projections.2.proj.bias": tensor(768),
            "decode_head.classifier.bias": tensor(2),
        }
        expected_keys = [
            "segformer.encoder.patch_embeddings.0.proj.weight",
            "segformer.encoder.patch_embeddings.0.layer_norm.weight",
            "segformer.encoder.block.1.2.layer_norm_1.bias",
            "segformer.encoder.block.1.2.attention.self.query.weight",
            "segformer.encoder.block.1.2.attention.self.sr.weight",
            "segformer.encoder.block.1.2.attention.self.layer_norm.weight",
            "segformer.encoder.block.1.2.attention.output.dense.bias",
            "segformer.encoder.block.1.2.layer_norm_2.weight",
            "segformer.encoder.block.1.2.mlp.dense1.weight",
            "segformer.encoder.block.1.2.mlp.dense2.bias",
            "segformer.encoder.layer_norm.3.bias",
            "decode_head.linear_c.2.proj.bias",
            "decode_head.classifier.bias",
        ]
        model_state = {
            key: torch.empty_like(value)
            for key, value in zip(expected_keys, checkpoint_state.values(), strict=True)
        }

        remapped = remap_segformer_state_dict_for_model(checkpoint_state, model_state)

        self.assertIsNot(remapped, checkpoint_state)
        self.assertEqual(set(remapped), set(expected_keys))
        self.assertEqual(remapped["segformer.encoder.block.1.2.mlp.dense1.weight"].shape, (512, 128))

    def test_remaps_segformer_encoder_checkpoint_to_stages_model_namespace(self) -> None:
        stages_state = {
            "segformer.stages.0.patch_embeddings.proj.bias": tensor(64),
            "segformer.stages.2.blocks.5.attention.v_proj.weight": tensor(320, 320),
            "segformer.stages.2.blocks.5.mlp.fc1.bias": tensor(1280),
            "decode_head.linear_projections.1.proj.weight": tensor(768, 128),
        }
        encoder_state = {
            "segformer.encoder.patch_embeddings.0.proj.bias": stages_state["segformer.stages.0.patch_embeddings.proj.bias"],
            "segformer.encoder.block.2.5.attention.self.value.weight": stages_state[
                "segformer.stages.2.blocks.5.attention.v_proj.weight"
            ],
            "segformer.encoder.block.2.5.mlp.dense1.bias": stages_state["segformer.stages.2.blocks.5.mlp.fc1.bias"],
            "decode_head.linear_c.1.proj.weight": stages_state["decode_head.linear_projections.1.proj.weight"],
        }

        remapped = remap_segformer_state_dict_for_model(encoder_state, stages_state)

        self.assertIsNot(remapped, encoder_state)
        self.assertEqual(set(remapped), set(stages_state))

    def test_does_not_remap_when_shapes_do_not_match(self) -> None:
        checkpoint_state = {"segformer.stages.0.patch_embeddings.proj.bias": tensor(64)}
        model_state = {"segformer.encoder.patch_embeddings.0.proj.bias": tensor(128)}

        remapped = remap_segformer_state_dict_for_model(checkpoint_state, model_state)

        self.assertIs(remapped, checkpoint_state)


class ResolveDeviceTest(unittest.TestCase):
    def test_explicit_device_string_is_honored(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        self.assertEqual(resolve_device("cuda:1"), torch.device("cuda:1"))

    def test_auto_picks_an_available_backend(self) -> None:
        device = resolve_device("auto")
        self.assertIn(device.type, {"cuda", "mps", "cpu"})
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            self.assertEqual(device.type, "cpu")


class ForwardLogitsTest(unittest.TestCase):
    def test_raw_tensor_output_is_interpolated_to_target(self) -> None:
        model = lambda images: torch.zeros(images.shape[0], 2, 8, 8)  # noqa: E731
        logits = forward_logits(model, torch.zeros(3, 3, 32, 32), (32, 32))
        self.assertEqual(tuple(logits.shape), (3, 2, 32, 32))

    def test_logits_attribute_output_is_unwrapped(self) -> None:
        model = lambda images: SimpleNamespace(logits=torch.ones(1, 2, 16, 16))  # noqa: E731
        logits = forward_logits(model, torch.zeros(1, 3, 16, 16), (16, 16))
        # already at target size: no interpolation, values untouched
        self.assertTrue(torch.equal(logits, torch.ones(1, 2, 16, 16)))

    def test_mask2former_query_output_is_projected_to_dense_logits(self) -> None:
        def model(images: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                class_queries_logits=torch.tensor([[[8.0, 0.0, -8.0], [0.0, 8.0, -8.0]]]),
                masks_queries_logits=torch.tensor([[[[-8.0, -8.0], [-8.0, -8.0]], [[8.0, 8.0], [8.0, 8.0]]]]),
            )

        logits = forward_logits(model, torch.zeros(1, 3, 8, 8), (8, 8))
        probs = torch.softmax(logits, dim=1)

        self.assertEqual(tuple(logits.shape), (1, 2, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(1, 8, 8), atol=1e-5))
        self.assertGreater(float(probs[:, 1].mean()), 0.99)


class LoadCheckpointTest(unittest.TestCase):
    def test_segformer_b3_eval_config_matches_mit_b3_shape(self) -> None:
        model = create_segformer_for_eval("segformer_b3")

        self.assertEqual(model.config.depths, [3, 4, 18, 3])
        self.assertEqual(model.config.hidden_sizes, [64, 128, 320, 512])
        self.assertEqual(model.config.decoder_hidden_size, 768)
        self.assertEqual(model.config.num_labels, 2)

    def test_resunet_checkpoint_round_trip(self) -> None:
        model = create_resunet(base_channels=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": {"model": "resunet", "base_channels": 4},
                    "epoch": 7,
                    "best_iou_sulfide": 0.93,
                },
                path,
            )
            loaded, meta = load_binary_segmentation_checkpoint(path, torch.device("cpu"))
        self.assertEqual(meta["model"], "resunet")
        self.assertEqual(meta["epoch"], 7)
        self.assertEqual(meta["best_iou_sulfide"], 0.93)
        self.assertEqual(meta["state_dict_compatibility"], "exact")
        original = model.state_dict()
        for key, value in loaded.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]), key)

    def test_unsupported_model_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            torch.save({"model": {}, "args": {"model": "unknown_arch"}}, path)
            with self.assertRaises(ValueError):
                load_binary_segmentation_checkpoint(path, torch.device("cpu"))

    def test_incompatible_state_dict_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mismatched.pt"
            torch.save(
                {
                    "model": {"stem.weight": tensor(1)},
                    "args": {"model": "resunet", "base_channels": 4},
                },
                path,
            )
            with self.assertRaises(RuntimeError) as ctx:
                load_binary_segmentation_checkpoint(path, torch.device("cpu"))
        self.assertIn(str(path), str(ctx.exception))
        self.assertIn("stem.weight", str(ctx.exception))


class CheckpointLoadErrorTest(unittest.TestCase):
    def test_message_names_path_first_key_and_original_error(self) -> None:
        message = _checkpoint_load_error(
            Path("/models/x.pt"),
            {"model": {"segformer.stages.0.w": tensor(1)}},
            RuntimeError("size mismatch"),
        )
        self.assertIn("/models/x.pt", message)
        self.assertIn("segformer.stages.0.w", message)
        self.assertIn("size mismatch", message)

    def test_empty_checkpoint_uses_placeholder_key(self) -> None:
        message = _checkpoint_load_error(Path("x.pt"), {}, RuntimeError("boom"))
        self.assertIn("<empty>", message)


if __name__ == "__main__":
    unittest.main()
