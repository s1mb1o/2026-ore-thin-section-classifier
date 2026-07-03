from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.model_io import remap_segformer_state_dict_for_model


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


if __name__ == "__main__":
    unittest.main()
