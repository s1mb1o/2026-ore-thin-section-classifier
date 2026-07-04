# Plan 39 — Pipeline resilience & recovery (graceful degradation)

- Date: 2026-07-04
- Status: draft (not implemented)
- Tracked as a Things Inbox task ("Implement ore-pipeline resilience & recovery").
- Motivated by the 2026-07-04 review of parallel-session work, which found the pipeline
  fails hard on several degradation modes. Two are confirmed HIGH bugs (see §5).
- Surfaces touched: `src/ore_classifier/resident_pipeline.py`,
  `scripts/run_resident_batch.py`, `scripts/infer_binary_sulfide.py`,
  `apps/ore_pipeline_web.py`, `scripts/build_grain_dataset.py`,
  `scripts/evaluate_official_pipeline.py`.

## 1. Goal

The pipeline must **degrade instead of crashing** and **resume instead of restarting**.
A single bad image, a model that won't load, an OOM on a huge panorama, or a full disk
should each produce a clear, recorded, recoverable outcome — never a lost batch or a
corrupt run directory.

## 2. Design principles

- **Fail soft** — when a model fails, fall back to the heuristic backend and mark the run
  `degraded`, don't fail it. A weak answer with provenance beats no answer.
- **Fail fast** — pre-flight the cheap checks (disk space, checkpoint validity, image
  readability) before spending GPU time.
- **Fail safe** — every persisted artifact is written atomically (`.tmp` → `rename`) so a
  crash can never leave a half-written `pipeline_summary.json` that poisons resume.
- **Observable** — the chosen backend, every fallback, and every skipped image are recorded
  in run provenance / `warnings`, so the report never silently misrepresents how a result
  was produced.
- **Bounded** — inference and subprocess calls have timeouts; OOM retries are capped.

## 3. Failure taxonomy → detection → recovery

| # | Failure | Detection | Degradation / recovery | Where |
| --- | --- | --- | --- | --- |
| F1 | ML model won't load (corrupt/missing ckpt, import error) | `validate_checkpoint` + load-time try/except | fall back to heuristic backend; run = `degraded`; provenance records fallback | `ore_pipeline_web.py`, `resident_pipeline.py` |
| F2 | ML inference raises mid-run (CUDA/MPS error, shape) | try/except around `forward_logits` | retry once → fall back to heuristic; never fail the whole run/batch | `resident_pipeline.py`, `run_ore_pipeline.py` |
| F3 | GPU/host OOM on large panorama | catch `torch.cuda.OutOfMemoryError` / MPS alloc / `MemoryError` | `empty_cache`, halve `batch_size`, then shrink `tile_size`, retry (capped); accumulators on **memmap** not RAM | `resident_pipeline.py`, `infer_binary_sulfide.py` |
| F4 | No space left on disk | pre-flight `shutil.disk_usage` vs estimated output; `OSError(ENOSPC)` on write | fail fast before inference with actionable message; on mid-write ENOSPC, clean up the partial run dir; atomic writes | all writers |
| F5 | Process/host crash mid-batch | on restart, `pipeline_summary.json` present per completed image | resume: skip complete images, re-run missing/partial ones; validate each per-image output before trusting it | `run_resident_batch.py` |
| F6 | One unreadable/garbage image | try/except on decode + inference | skip that image, record in manifest, continue (`--keep-going`); do not abort batch | `run_resident_batch.py`, `build_grain_dataset.py` |
| F7 | Web job hangs / long inference | daemon threads (exists) + per-job timeout/watchdog | cancel the job, mark `failed` with reason `timeout`; server stays responsive | `ore_pipeline_web.py` |
| F8 | Talc ML fails specifically | subprocess return code / exception | fall back to the heuristic talc candidate (already implemented) rather than failing the run; record talc backend used | `ore_pipeline_web.py`, `run_ore_pipeline.py` |

## 4. Recovery-state model

- **Run states**: extend the existing set with `degraded` (completed via fallback) distinct
  from `complete` (completed as configured) and `failed` (no usable result). `degraded`
  must be visible in the UI/report and in `ore_summary.json` warnings.
- **Atomic run dirs**: write each artifact to `<name>.tmp` then `os.replace`; write
  `pipeline_summary.json` **last** (already the case — keep it) so its presence means
  "image done". A run dir without the summary is presumed partial and re-run on resume.
- **Idempotent resume**: `run_resident_batch.py` already resumes on `pipeline_summary.json`;
  harden it to (a) verify the referenced mask/overlay files exist and are non-empty and
  (b) re-run the image if any are missing/corrupt, rather than trusting the summary alone.
- **Batch exit codes**: `0` = all done; `3` = completed with some images skipped/degraded
  (not fatal); `2` = fatal (config/IO). Downstream (`evaluate_official_pipeline.py
  --resident`) must treat `3` as success-with-warnings, not abort. (Current code returns
  `2` on any per-image failure even under `--keep-going` — fix.)

## 4a. Per-run degradation record (MANDATORY)

Every degradation must be recorded **on the specific run it happened to**, so anyone
reading a result knows it may be sub-optimal. A degradation that is recovered but not
recorded is still a bug.

- Each run summary (`pipeline_summary.json` / `ore_summary.json` / the grain
  `dataset_summary.json`) carries a machine-readable **`degradations: []`** list. Empty
  list ⇒ the run ran exactly as configured.
- Every recovery event appends a structured entry, e.g.
  `{ "code": "model_fallback_heuristic", "detail": "talc ckpt load failed: <err>",
  "severity": "warning", "at": "<iso>" }`. Codes cover at least: `model_fallback_heuristic`,
  `oom_batch_shrunk`, `oom_tile_shrunk`, `image_skipped`, `grain_skipped_bad_bbox`,
  `output_truncated`, `resume_reran_partial`.
- A non-empty list sets a top-level `result_quality: "degraded"` (vs `"nominal"`) and
  `needs_expert_review = true`, surfaced in the UI result panel and the PDF report — the
  run must visibly announce "this result may be sub-optimal", never hide it.
- Aggregations (history stats, eval harness) must be able to filter or flag `degraded`
  runs so a degraded result is never silently averaged in as a nominal one.

**Already landed toward this (2026-07-04):**
- `scripts/build_grain_dataset.py` records `grains_skipped_bad_bbox` in
  `dataset_summary.json` (F6 grain-skip counted per build, not lost).
- `src/ore_classifier/resident_pipeline.py` now emits a `degradations: []` list plus
  `result_quality: nominal|degraded` on both the sulfide `summary.json` and the run
  `pipeline_summary.json`, covering F2 (sulfide model failure → brightness-heuristic
  fallback), F3 (OOM → adaptive batch shrink via `_accumulate_prob_map`, memmap-backed),
  and talc-model failure → heuristic-candidate fallback. Unit-tested in
  `tests/test_resident_resilience.py`.

**Remaining surface:** the web runtime (`apps/ore_pipeline_web.py`) still needs the same
`degradations`/`result_quality` wiring and the F7 job watchdog; deferred while that file
is held by another active session.

## 5. Confirmed HIGH bugs to fix first (from the review)

1. **`resident_pipeline.py:85-86`** — `prob_sum`/`weight_sum` are in-RAM `np.zeros` float32;
   the reference `infer_binary_sulfide.py:67-68` uses `np.memmap`. On 27025×21227 panoramas
   this OOMs a long-lived resident process. Swap to memmap (F3). This is the concrete
   instance of the "no space / no memory" degradation the user called out.
2. **`build_grain_dataset.py:140`** — `grains_skipped_bad_bbox += 1` on an uninitialized
   counter → `UnboundLocalError` in the `except` meant to skip bad grains, aborting the
   build (F6). Initialize the counter (and add it to `summary`).

## 6. Implementation tasks (ordered)

1. Fix the two HIGH bugs (§5) — smallest, highest value.
2. Add a `resilience` config block (thresholds/knobs, §7) and a `degraded` run state.
3. Wrap model load + inference (sulfide, talc) with the F1/F2/F8 fallback-to-heuristic
   path; record backend + fallback reason in provenance/warnings.
4. Add the F3 OOM handler: catch OOM, `empty_cache`, adaptively shrink `batch_size` then
   `tile_size`, retry up to N times, keep memmap accumulators.
5. Add F4 pre-flight disk check + atomic writes everywhere the pipeline persists; clean up
   partial run dirs on failure.
6. Harden resume (F5) and fix batch exit-code semantics (F6 / §4).
7. Add F7 per-job timeout/watchdog to the web runtime.
8. Fault-injection tests (§8).
9. Docs: update `docs/session-sync.md`, `ChangeLog.md`, `SMOKE_TESTS.md`; model cards note
   the fallback behavior.

## 7. Config knobs (proposed)

```text
resilience:
  min_free_disk_mb: 2048            # F4 pre-flight
  oom_max_retries: 3                # F3
  oom_batch_shrink: 0.5             # halve on each OOM
  oom_min_tile: 768                 # floor for tile shrink
  job_timeout_s: 1800               # F7 web job watchdog
  on_model_failure: fallback_heuristic   # fallback_heuristic | fail
  keep_going: true                  # F6 batch continues past a bad image
```

Default `on_model_failure: fallback_heuristic` — but expose `fail` for eval runs where a
silent heuristic fallback would corrupt a benchmark (we must not report heuristic numbers
as ML numbers). The eval harness should set `fail` so degradation is never mistaken for a
model result.

## 8. Testing / fault injection

- Corrupt checkpoint file → expect `degraded` run via heuristic, provenance records it.
- Monkeypatch the model forward to raise `torch.cuda.OutOfMemoryError` → expect batch
  shrink + retry, then heuristic fallback if still failing.
- `tmp_path` on a tiny filesystem / patch `shutil.disk_usage` to near-zero → expect
  fast pre-flight failure, no partial run dir left behind.
- Kill `run_resident_batch.py` after K images (send SIGTERM) → restart → expect only the
  remaining images run, prior results intact.
- Feed a truncated/zero-byte image in a batch → expect skip + manifest entry + exit code 3.

## 9. Non-goals / open questions

- Not building distributed/multi-host failover — single-host resilience only (sharding
  across gx10/zelda stays a manual op per plan 26).
- Open: should `degraded` runs be excluded from history-level aggregate stats by default?
- Open: disk-retention/auto-prune of old `outputs/ore_pipeline_ui/runs/` — needed for F4
  on long-lived servers, but deletion policy needs user sign-off (don't auto-delete runs).
- Open: for the web runtime, is `fallback_heuristic` the right default for a *demo*, or
  should the jury-facing demo surface a visible "model failed → heuristic" banner?
