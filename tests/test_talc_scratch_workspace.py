from __future__ import annotations

import base64
import io
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from apps.talc_review_web import TalcReviewStore  # noqa: E402
from ore_classifier.talc_blue_line_converter import read_mask  # noqa: E402
from prepare_talc_scratch_workspace import build_workspace, collect_images  # noqa: E402


def mask_data_url(mask: np.ndarray) -> str:
    handle = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(handle, format="PNG")
    return "data:image/png;base64," + base64.b64encode(handle.getvalue()).decode("ascii")


class TalcScratchWorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "outputs/test_talc_scratch_workspace"
        shutil.rmtree(self.root, ignore_errors=True)
        self.class_a = self.root / "class_a"
        self.class_b = self.root / "class_b"
        self.workspace = self.root / "workspace"
        self.class_a.mkdir(parents=True, exist_ok=True)
        self.class_b.mkdir(parents=True, exist_ok=True)
        rgb_a = np.full((60, 80, 3), (60, 70, 50), dtype=np.uint8)
        rgb_b = np.full((50, 70, 3), (30, 30, 35), dtype=np.uint8)
        Image.fromarray(rgb_a, mode="RGB").save(self.class_a / "img_one.JPG")
        Image.fromarray(rgb_b, mode="RGB").save(self.class_b / "img_two.JPG")
        # Same stem in another folder must get a unique sample id.
        Image.fromarray(rgb_b, mode="RGB").save(self.class_b / "img_one.JPG")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self) -> dict:
        images = collect_images([self.class_a, self.class_b], per_dir_limit=None, shuffle_seed=None)
        return build_workspace(images, self.workspace)

    def test_builds_manifest_with_empty_masks_and_unique_ids(self) -> None:
        manifest = self.build()
        self.assertEqual(manifest["sample_count"], 3)
        ids = [sample["image_id"] for sample in manifest["samples"]]
        self.assertEqual(len(ids), len(set(ids)))
        for sample in manifest["samples"]:
            final_mask = read_mask(Path(sample["paths"]["final_talc_mask"]))
            self.assertEqual(int(np.count_nonzero(final_mask)), 0)
            self.assertEqual(final_mask.shape, (sample["height"], sample["width"]))
            self.assertTrue(Path(sample["paths"]["source_image"]).exists())
            self.assertTrue(Path(sample["original_path"]).exists())
            self.assertEqual(sample["status"], "scratch_unlabeled")

    def test_review_store_loads_scratch_workspace_as_editable(self) -> None:
        self.build()
        store = TalcReviewStore(
            annotated_dir=None,
            original_dir=None,
            workspace_dir=self.workspace,
            conversion_dir=self.workspace,
            sulfide_mask_dir=None,
            silicate_mask_dir=None,
            reconvert=False,
            limit=None,
            sam2_model_id="facebook/sam2.1-hiera-tiny",
            sam2_device="cpu",
        )
        manifest = store.manifest_payload()
        self.assertEqual(manifest["sample_count"], 3)
        for card in manifest["samples"]:
            self.assertTrue(card["has_original"], card)
        sample_id = manifest["samples"][0]["sample_id"]
        payload = store.sample_payload(sample_id)
        self.assertTrue(payload["editable"])
        self.assertEqual(payload["metrics"]["current_talc_pixels"], 0)

    def test_save_reviewed_mask_roundtrip_on_scratch_sample(self) -> None:
        self.build()
        store = TalcReviewStore(
            annotated_dir=None,
            original_dir=None,
            workspace_dir=self.workspace,
            conversion_dir=self.workspace,
            sulfide_mask_dir=None,
            silicate_mask_dir=None,
            reconvert=False,
            limit=None,
            sam2_model_id="facebook/sam2.1-hiera-tiny",
            sam2_device="cpu",
        )
        manifest = store.manifest_payload()
        sample_id = manifest["samples"][0]["sample_id"]
        payload = store.sample_payload(sample_id)
        height = int(payload["image"]["height"])
        width = int(payload["image"]["width"])
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[5:20, 10:40] = 255
        result = store.save_current_mask(
            sample_id,
            {
                "mask_png": mask_data_url(mask),
                "edits": [{"type": "lasso_add_talc", "point_count": 42}],
                "reviewer": "unit-test",
            },
            reviewed=True,
        )
        self.assertTrue(result["reviewed"])
        reviewed_path = Path(result["review_summary"]["paths"]["reviewed_talc_mask"])
        self.assertTrue(reviewed_path.exists())
        saved = read_mask(reviewed_path)
        self.assertEqual(int(np.count_nonzero(saved)), int(np.count_nonzero(mask)))
        patch = json.loads(Path(result["review_summary"]["paths"]["review_patch"]).read_text(encoding="utf-8"))
        self.assertEqual(patch["edits"][0]["type"], "lasso_add_talc")


if __name__ == "__main__":
    unittest.main()
