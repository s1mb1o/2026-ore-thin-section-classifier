"""Resilience unit tests for the resident pipeline's tiled accumulation helper.

Exercises the OOM-retry / degradation logic of ``_accumulate_prob_map`` and the
OOM classifier ``_is_oom_error`` without loading any model (a fake ``forward_fn``
stands in for the network), per docs/plans/39_pipeline-resilience-and-recovery.md.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ore_classifier.resident_pipeline import (  # noqa: E402
    _accumulate_prob_map,
    _atomic_write_text,
    _is_oom_error,
)
from ore_classifier.tiling import iter_tiles  # noqa: E402
import run_resident_batch as rrb  # noqa: E402

_TILE = 2
_DIM = 4


def _tiles():
    return iter_tiles(width=_DIM, height=_DIM, tile_size=_TILE, stride=_TILE)


def _weight():
    return np.ones((_TILE, _TILE), dtype=np.float32)


class TestOomClassifier(unittest.TestCase):
    def test_recognizes_oom_and_ignores_other_errors(self):
        self.assertTrue(_is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate...")))
        self.assertTrue(_is_oom_error(MemoryError()))
        self.assertFalse(_is_oom_error(ValueError("bad input")))
        self.assertFalse(_is_oom_error(RuntimeError("shape mismatch")))


class TestAccumulateProbMap(unittest.TestCase):
    def test_clean_run_records_no_degradation(self):
        tiles = _tiles()
        calls = {"n": 0}

        def fwd(batch_tiles):
            calls["n"] += 1
            return np.full((len(batch_tiles), _TILE, _TILE), 0.8, dtype=np.float32)

        with tempfile.TemporaryDirectory() as out_dir:
            prob, processed, batch, degr = _accumulate_prob_map(
                forward_fn=fwd, tiles=tiles, weight=_weight(), width=_DIM, height=_DIM,
                batch_size=4, out_dir=Path(out_dir),
            )
        self.assertEqual(processed, len(tiles))
        self.assertEqual(degr, [])
        self.assertEqual(batch, 4)
        self.assertEqual(prob.shape, (_DIM, _DIM))
        self.assertTrue(np.allclose(prob, 0.8, atol=1e-5))

    def test_oom_shrinks_batch_and_recovers(self):
        tiles = _tiles()
        state = {"n": 0}

        def fwd(batch_tiles):
            state["n"] += 1
            if state["n"] == 1:  # fail the first (full-batch) attempt only
                raise RuntimeError("CUDA out of memory")
            return np.full((len(batch_tiles), _TILE, _TILE), 0.5, dtype=np.float32)

        with tempfile.TemporaryDirectory() as out_dir:
            prob, processed, batch, degr = _accumulate_prob_map(
                forward_fn=fwd, tiles=tiles, weight=_weight(), width=_DIM, height=_DIM,
                batch_size=4, out_dir=Path(out_dir),
            )
        self.assertEqual(len(degr), 1)
        self.assertEqual(degr[0]["code"], "oom_batch_shrunk")
        self.assertEqual(batch, 2)
        self.assertEqual(processed, len(tiles))
        self.assertTrue(np.allclose(prob, 0.5, atol=1e-5))

    def test_persistent_oom_propagates_after_retries(self):
        tiles = _tiles()

        def fwd(batch_tiles):
            raise RuntimeError("CUDA out of memory")

        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaises(RuntimeError):
                _accumulate_prob_map(
                    forward_fn=fwd, tiles=tiles, weight=_weight(), width=_DIM, height=_DIM,
                    batch_size=2, out_dir=Path(out_dir), oom_max_retries=2,
                )

    def test_non_oom_error_is_not_retried(self):
        tiles = _tiles()
        state = {"n": 0}

        def fwd(batch_tiles):
            state["n"] += 1
            raise ValueError("corrupt weights")

        with tempfile.TemporaryDirectory() as out_dir:
            with self.assertRaises(ValueError):
                _accumulate_prob_map(
                    forward_fn=fwd, tiles=tiles, weight=_weight(), width=_DIM, height=_DIM,
                    batch_size=4, out_dir=Path(out_dir),
                )
        self.assertEqual(state["n"], 1)  # no retry on a non-OOM failure


class TestAtomicWrite(unittest.TestCase):
    def test_replaces_and_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pipeline_summary.json"
            _atomic_write_text(path, "first\n")
            self.assertEqual(path.read_text(), "first\n")
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            _atomic_write_text(path, "second\n")  # overwrite is atomic too
            self.assertEqual(path.read_text(), "second\n")


class TestRunIsComplete(unittest.TestCase):
    def _make_run(self, root, *, with_summary=True, valid_json=True, artifacts=True):
        run_dir = Path(root) / "runs" / "row_ore" / "r1"
        (run_dir / "binary_sulfide").mkdir(parents=True, exist_ok=True)
        (run_dir / "ore_analysis").mkdir(parents=True, exist_ok=True)
        sulfide = run_dir / "binary_sulfide" / "sulfide_mask.png"
        ore = run_dir / "ore_analysis" / "ore_summary.json"
        comp = run_dir / "ore_analysis" / "component_features.csv"
        if artifacts:
            sulfide.write_bytes(b"\x89PNG")
            ore.write_text("{}")
            comp.write_text("component_id\n")
        if with_summary:
            body = {"paths": {"sulfide_mask": str(sulfide), "ore_summary": str(ore), "component_features": str(comp)}}
            (run_dir / "pipeline_summary.json").write_text(json.dumps(body) if valid_json else "{not json")
        return run_dir

    def test_complete_run_is_trusted(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(rrb._run_is_complete(self._make_run(d)))

    def test_missing_summary_reruns(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(rrb._run_is_complete(self._make_run(d, with_summary=False)))

    def test_corrupt_summary_reruns(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(rrb._run_is_complete(self._make_run(d, valid_json=False)))

    def test_missing_artifact_reruns(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(rrb._run_is_complete(self._make_run(d, artifacts=False)))


if __name__ == "__main__":
    unittest.main()
