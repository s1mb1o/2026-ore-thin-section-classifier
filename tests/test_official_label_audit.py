from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_official_labels import build_label_audit  # noqa: E402


class OfficialLabelAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "outputs/test_official_label_audit"
        shutil.rmtree(self.root, ignore_errors=True)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir(parents=True)
        Image.new("RGB", (8, 8), (100, 120, 90)).save(self.dataset / "same_a.jpg")
        shutil.copy2(self.dataset / "same_a.jpg", self.dataset / "same_b.jpg")
        Image.new("RGB", (8, 8), (10, 10, 10)).save(self.dataset / "unique.jpg")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_detects_duplicate_content_with_conflicting_labels(self) -> None:
        items = [
            {"path": "same_a.jpg", "label_hint": "talcose", "width": 8, "height": 8, "bytes": 1},
            {"path": "same_b.jpg", "label_hint": "ordinary_intergrowth", "width": 8, "height": 8, "bytes": 1},
            {"path": "unique.jpg", "label_hint": "fine_intergrowth", "width": 8, "height": 8, "bytes": 1},
        ]

        audit = build_label_audit(items, dataset_root=self.dataset)

        self.assertEqual(audit["duplicate_group_count"], 1)
        self.assertEqual(audit["label_conflict_group_count"], 1)
        self.assertEqual(set(audit["conflict_paths"]), {"same_a.jpg", "same_b.jpg"})


if __name__ == "__main__":
    unittest.main()
