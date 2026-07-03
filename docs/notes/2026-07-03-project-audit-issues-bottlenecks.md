# Project Audit — Issues, Bottlenecks, Low-Hanging Fruit

Date: 2026-07-03
Scope audited: 76 source files (~30.5k LOC excl. `.venv`/`outputs`/`dataset`), 27 test modules,
the two monolithic web apps, docker runtime, and the docs/plans corpus.

Findings are ranked by impact. File references use `path:line`. This note records the state at
audit time; several items are being fixed in the same session (see the Status column).

## Status of fixes (this session)

| # | Finding | Action taken |
|---|---------|--------------|
| 1 | `.venv` was Python 3.9.6 vs required 3.10+ | Rebuilt `.venv` on Python 3.12; added `requirements.lock` |
| 2 | `transformers>=5.12.1` contradicted verified 5.5.4 | requirements.txt reconciled to `>=5.5.4`; exact pins in `requirements.lock` |
| 4 | 10.9k-line `ore_pipeline_web.py`, ~5.7k lines inline HTML/JS | `HTML_PAGE` extracted to `apps/static/ore_pipeline_ui.html`, served from disk |
| 7 | 4.2 GB `outputs/` with 25 disposable smoke/probe dirs | Added `scripts/clean_smoke_outputs.sh` |
| — | Performance bottlenecks | Separate pass; see the perf section below and ChangeLog |

Deferred (documented, not changed this session): #3 CI, #5 VM redeploy, #6 request-thread CPU
work, #8 public-exposure auth, and the talc-model-wiring / metric-compliance gaps.

**Decision (2026-07-04) on full-resolution preprocessing (#6):** measured ~12 s on a ~24 MP
image with the default preset (denoise ON), vs ~1.5 s at 2600 px analysis scale; denoise alone
is ~6.5 s at 24 MP. Because the non-deferred path feeds analysis via `downscale(apply_preprocessing(full))`,
speeding it up either changes analysis results (preprocess at analysis scale, matching the panorama
path) or needs a background-worker refactor of the synchronous handler. The owner chose to **leave
the code as-is** and instead **recommend disabling denoise for large images** during demos. The
only code change made was a provably pixel-identical cleanup of `apply_preprocessing` (single numpy
buffer, no per-step PIL↔numpy round-trips) that lowers peak memory without touching outputs.

## 🔴 Critical / correctness

1. **`.venv` was Python 3.9.6, code requires 3.10+.**
   `src/ore_classifier/segmentation_metrics.py:47` uses `zip(..., strict=True)`; 6 files use
   `strict=True` and 21 use PEP 604 `X | None` annotations. `.venv/bin/python -m unittest`
   errored with `zip() takes no keyword arguments`. `COMMANDS.md:30` already said to recreate the
   venv on >=3.10, so the committed venv was stale and also lacked `transformers`. Note system
   `python3` is now 3.14.4 (torch/numpy wheels may lag) — pinned the venv to 3.12.

2. **`requirements.txt` demanded a transformers version never verified.**
   It pinned `transformers>=5.12.1`, but `docs/session-sync.md:162` records the only verified local
   runtime as `torch 2.11.0` + `transformers 5.5.4` (below the floor, so it would not satisfy the
   pin). Reconciled to the verified versions plus a `requirements.lock` frozen from a known-good env.

3. **No CI, no dependency pinning.** No `.github/workflows`, `Makefile`, `tox`, or `nox`;
   `requirements.txt` was fully unpinned. A minimal GitHub Action running `unittest` on py3.12 plus
   a lockfile would have caught #1/#2 automatically. (Not added this session.)

## 🟠 Bottlenecks

4. **`apps/ore_pipeline_web.py` = 10,892 lines / 565 KB, of which the single `HTML_PAGE` raw
   string spans lines 5214–10876 (~5,700 lines of HTML/CSS/JS).** `talc_review_web.py` is another
   4,453 lines the same way. The JS got no linting, highlighting, or tests. `HTML_PAGE` was served
   verbatim (`ore_pipeline_web.py:5169 return HTML_PAGE`) with no interpolation, so it was extracted
   to `apps/static/ore_pipeline_ui.html` and is now read from disk. `talc_review_web.py` uses an
   f-string (`render_html_page()` at `:898`) with interpolation, so it is left as a follow-up.

5. **Panorama runs are CPU-bound on synchronous full-size `preprocessed_full.png`.**
   `docs/session-sync.md:92` flags the public VM path as not demo-ready until optimized. The local
   defer-fix exists but the running VM container is still the old image (`session-sync.md:152`).

6. **`ThreadingHTTPServer` + heavy synchronous CPU work inside request handlers**
   (`ore_pipeline_web.py:5130`). Under the GIL, concurrent large-image requests serialize on
   numpy/PIL/cv2 work and compete with `/status` polling. Fine for single-user demo, a wall for
   concurrent judging.

7. **`outputs/` = 4.2 GB, `models/` = 1.2 GB in the working tree.** 25 `*smoke*`/`*probe*` dirs
   are disposable (e.g. `smoke_train_segformer_b2` 567 MB). `outputs/ore_pipeline_ui` is 1.4 GB of
   run history. All gitignored, but they bloat backups and slow file tooling. Added
   `scripts/clean_smoke_outputs.sh`.

## 🟡 Security (matters because it is publicly exposed)

8. **Public Docker VM binds `0.0.0.0:8080` with zero auth** (`docker/ore-pipeline-ui/entrypoint.sh:13`,
   live at `http://111.88.145.15:8080`). Exposes file upload, a REST API, a server-side path loader
   (`register_upload_from_path`, `ore_pipeline_web.py:1702`), and a subprocess checkpoint probe
   (`:2401`). Artifact serving is guarded by `is_relative_to` (`:198`) — good — but recommend a
   reverse proxy + basic auth / shared token, and confirming `register_upload_from_path` cannot be
   driven with an attacker-chosen absolute path when exposed. May be an accepted risk for a private
   demo window — flagging so it is a decision, not an accident.

## What we are missing (gaps, not bugs)

- **Talc production path is still broken.** `docs/session-sync.md:106`: the production HSV talc
  candidate scores IoU 0.000 vs reviewed GT; a trained SegFormer-B0 reaches 0.644 but is not wired
  into `run_ore_pipeline.py`/the UI as the default talc source. The demo ships a talc detector known
  to be worthless. Biggest correctness gap in the actual pipeline.
- **Official metric compliance incomplete.** Organizers want IoU + Hausdorff (seg), F1 + AUC
  (classification); `session-sync.md:149` admits current weak-label IoU benchmarks are incomplete by
  that standard. No panorama-compliance benchmark run exists.
- **No end-to-end test that executes an ML run.** Tests use the heuristic backend; the ML path
  (checkpoint load, tiled inference, namespace remap) is only manually smoke-verified.
- **Rule calibration is stale.** `session-sync.md:106`: calibrations that consumed HSV `talc_fraction`
  need regeneration, since they were built on the broken talc signal.

## Recommended order

1. Env/deps/CI trio (#1/#2/#3) — unblocks trustworthy testing. (venv + lock done; CI still open.)
2. Wire the trained talc model in and regenerate calibration — the demo currently claims talc
   detection it cannot back.
3. Redeploy the VM image so the panorama defer-fix is live (#5).
4. Extract the remaining HTML monolith (`talc_review_web.py`) before more UI work.
