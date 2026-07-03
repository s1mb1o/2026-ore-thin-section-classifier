from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from merge_official_batch_shards import merge_shards  # noqa: E402


class MergeOfficialBatchShardsTest(unittest.TestCase):
    def test_merge_shards_writes_combined_summary_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "ordinary_intergrowth"
            shard_b = root / "talcose"
            write_summary(shard_a, [{"run_id": "a", "source_label": "ordinary_intergrowth", "predicted_ore_class": "row_ore"}])
            write_summary(shard_b, [{"run_id": "b", "source_label": "talcose", "predicted_ore_class": "talcose_ore"}])
            (shard_b / "failures.json").write_text(
                json.dumps([{"source_rel_path": "bad.jpg", "error": "boom"}]),
                encoding="utf-8",
            )

            result = merge_shards([shard_a, shard_b], out_dir=root / "combined")

            with (root / "combined/summary.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            failures = json.loads((root / "combined/failures.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "combined/summary.json").read_text(encoding="utf-8"))

        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["failures"], 1)
        self.assertEqual([row["run_id"] for row in rows], ["a", "b"])
        self.assertTrue(rows[0]["shard_dir"].endswith("ordinary_intergrowth"))
        self.assertEqual(failures[0]["source_rel_path"], "bad.jpg")
        self.assertEqual(summary["shards"][1]["failures"], 1)

    def test_duplicate_run_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "a"
            shard_b = root / "b"
            write_summary(shard_a, [{"run_id": "same", "source_label": "ordinary_intergrowth"}])
            write_summary(shard_b, [{"run_id": "same", "source_label": "fine_intergrowth"}])

            with self.assertRaisesRegex(ValueError, "duplicate run_id"):
                merge_shards([shard_a, shard_b], out_dir=root / "combined")


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    (path / "failures.json").write_text("[]\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
