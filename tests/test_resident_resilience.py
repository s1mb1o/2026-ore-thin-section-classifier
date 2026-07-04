"""Resilience unit tests for the resident pipeline's tiled accumulation helper.

Exercises the OOM-retry / degradation logic of ``_accumulate_prob_map`` and the
OOM classifier ``_is_oom_error`` without loading any model (a fake ``forward_fn``
stands in for the network), per docs/plans/39_pipeline-resilience-and-recovery.md.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.resident_pipeline import (  # noqa: E402
    _accumulate_prob_map,
    _is_oom_error,
)
from ore_classifier.tiling import iter_tiles  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
