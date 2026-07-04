# Session Activity (live coordination)

**Purpose.** A live, ephemeral registry of what each concurrent agent/chat session is
doing *right now*, so parallel sessions do not collide on the same files, duplicate work,
or reuse an in-flight `docs/plans/NN_` / `docs/specs/` number.

This is **not** the durable handoff — that stays in `docs/session-sync.md`. This file is
throwaway coordination state: entries are added when work starts, refreshed as focus
shifts, and marked `done` (or pruned) when a session finishes or hands off.

## Protocol

1. **Before** starting meaningful work, read this file. If another `active` entry claims
   files or a plan/spec number you were about to touch, coordinate — pick different files,
   a different plan number, or wait.
2. Add/refresh your own entry with: last-updated ISO timestamp, a short session label,
   what you are doing now, the files/dirs you are actively editing, plan/spec names you
   are claiming, and status (`active` / `blocked` / `done`).
3. On finish or handoff: set status `done` and move durable outcomes into
   `docs/session-sync.md` + `ChangeLog.md`. Prune stale `done` rows when convenient.
4. Timestamps are local ISO 8601 (e.g. `date "+%Y-%m-%dT%H:%M:%S%z"`).

## Active / recent sessions

| Updated | Session | Doing now | Files / areas claimed | Status |
| --- | --- | --- | --- | --- |
| 2026-07-04T10:37:32+0300 | ore-overlay-class-percentages | Adding class percentages to the ore pipeline left/right segmentation legend overlays. Avoiding talc review files claimed by active session. | `apps/static/ore_pipeline_ui.html`, focused `tests/test_ore_pipeline_web.py` if needed; shared docs only if no conflict | active |
| 2026-07-04T10:40:48+0300 | talc-threshold-widget | Done: added confirmed Talc >=10% visible-pixel target display plus live Positive bag/Talc percentages to the talc review overlay widget. | `apps/talc_review_web.py`, `tests/test_talc_review_web.py`, `tests/browser/test_talc_review_ui_browser.py`, `docs/ui/v2/specs/talc-mask-review-web-app-v0.1.md`, `SMOKE_TESTS.md`, `ChangeLog.md`, `docs/session-sync.md`, `docs/session-activity.md` | done |
| 2026-07-04T10:15:07+0300 | resilience-batch | Done (uncommitted): plan 39 batch/resident resilience — exit-code semantics (F6) in `run_resident_batch.py`; resident OOM adaptive-batch retry (F3) + sulfide/talc model→heuristic fallback (F2) + per-run `degradations`/`result_quality` in `resident_pipeline.py`; unit test `tests/test_resident_resilience.py` (5 pass). Did NOT touch web app (held by ore-grain-reporting-impl). | `scripts/run_resident_batch.py`, `src/ore_classifier/resident_pipeline.py`, `tests/test_resident_resilience.py`, `docs/plans/39_pipeline-resilience-and-recovery.md` | done |
| 2026-07-04T10:12:30+0300 | gis-export-feature-doc | Done: created a focused document listing implemented GIS export features. README link skipped because the index already has unrelated uncommitted edits. | `docs/ui/v2/notes/2026-07-04-gis-export-implemented-features.md` | done |
| 2026-07-04T10:24:00+0300 | ore-grain-reporting-impl | Done: implemented Plan 42 grain-reporting API/UI fields, focused tests, docs handoff, and verified committed branch state. | `src/ore_classifier/component_analysis.py`, `apps/ore_pipeline_web.py`, `apps/static/ore_pipeline_ui.html`, `tests/test_ore_pipeline_grain_reporting.py`, `tests/test_ore_pipeline_web.py`, `docs/ui/v2/specs/ore-pipeline-grain-reporting-v0.1.md`, `docs/ui/v2/plans/42_ore-pipeline-grain-reporting.md`, `ChangeLog.md`, `SMOKE_TESTS.md`, `docs/session-sync.md` | done |
| 2026-07-04T10:00:50+0300 | gis-export-content-validation | Done: tightened Shapefile/GeoJSON export-content validation in focused GIS tests. | `tests/test_gis_export.py` | done |
| 2026-07-04T10:02:45+0300 | talc-cluster-overlay | Done: added tunable display-only talc-cluster overlay to the talc annotation browser tool, with focused unit/browser checks and docs. | `apps/talc_review_web.py`, `tests/test_talc_review_web.py`, `tests/browser/test_talc_review_ui_browser.py`, `docs/ui/v2/specs/talc-mask-review-web-app-v0.1.md`, `SMOKE_TESTS.md`, `ChangeLog.md`, `docs/session-sync.md`, `docs/session-activity.md` | done |
| 2026-07-04T08:40:20+0300 | grain-reporting-scope-fix | Done: removed remaining SEM/BSE/EDS and mineral-fraction wording from the ore-grain reporting draft; scope is strictly current OM masks/component features. | `docs/ui/v2/specs/ore-pipeline-grain-reporting-v0.1.md`, `docs/ui/v2/plans/42_ore-pipeline-grain-reporting.md` | done |
| 2026-07-04T08:39:43+0300 | cli-batch-doc | Done: documented how the v2 REST/UI pipeline can be run as a CLI batch workflow and where the CLI/API boundary is. Shared handoff docs left untouched because active sessions claim them. | `docs/ui/v2/notes/2026-07-04-ore-pipeline-cli-batch-contract.md`, `docs/ui/v2/README.md` | done |
| 2026-07-04T08:37:29+0300 | gis-export-tests | Done: added focused regression coverage for completed-run GIS artifacts in metadata, file listing, and artifact ZIP without touching shared web tests. | `tests/test_gis_export.py` | done |
| 2026-07-04T08:35:44+0300 | ore-grain-reporting-ui | Paused after reviewing existing grain-reporting code; added only the focused grain-reporting spec/plan draft, no implementation code changed. | `docs/ui/v2/specs/ore-pipeline-grain-reporting-v0.1.md`, `docs/ui/v2/plans/42_ore-pipeline-grain-reporting.md`, code reviewed in `src/ore_classifier/component_analysis.py`, `src/ore_classifier/component_reports.py`, `apps/ore_pipeline_web.py`, `apps/static/ore_pipeline_ui.html`, `tests/test_ore_pipeline_web.py` | done |
| 2026-07-04T08:06:26+0300 | gis-export-vectors | Done: added v0.1 GIS export spec/plan plus GeoJSON and Shapefile run artifacts with focused tests. Shared handoff docs left untouched because active sessions claim them. | `docs/ui/v2/specs/ore-pipeline-gis-export-v0.1.md`, `docs/ui/v2/plans/41_ore-pipeline-gis-export.md`, `src/ore_classifier/gis_export.py`, `apps/ore_pipeline_web.py`, `tests/test_gis_export.py` | done |
| 2026-07-04T07:56:44+0300 | panorama-compliance-card-plan | Done: created v2 UI plan 40 for the real panorama compliance card and linked it from the UI docs index/TODO backlog. Shared `ChangeLog.md`/`docs/session-sync.md` left untouched because active sessions still claim them. | `docs/ui/v2/plans/40_ore-pipeline-panorama-compliance-card.md`, `docs/ui/v2/README.md`, `docs/ui/v2/TODO_CANDIDATES.md` | done |
| 2026-07-04T08:36:16+0300 | ore-ui-run-tech-details | Done: added run technical details widget after sulfide-grain table, updated focused docs/tests, and restarted local UI on `127.0.0.1:63589`. | `apps/static/ore_pipeline_ui.html`, `tests/test_ore_pipeline_web.py`, `docs/ui/v2/ore-pipeline-ui-user-guide.md`, `docs/ui/v2/ore-pipeline-ui-customer-journeys.md`, shared `ChangeLog.md`/`SMOKE_TESTS.md`/`docs/session-sync.md` | done |
| 2026-07-04T07:50:36+0300 | ideas-current-state-review | Done: reviewed ideas against current v2 UI/model/project state; updated focused idea/backlog docs and `ResearchLog.md`. Shared `ChangeLog.md`/`docs/session-sync.md` were left untouched because active sessions claim them. | `docs/notes/2026-07-04-current-state-ideas-review.md`, `docs/notes/2026-07-03-pipeline-improvement-proposals.md`, `docs/ui/v2/TODO_CANDIDATES.md`, `ResearchLog.md` | done |
| 2026-07-04T07:54:15+0300 | mlflow-debug-tracking | Done: added optional MLflow debug tracking to the four train scripts via a shared no-op-by-default helper. | `src/ore_classifier/tracking.py`, `scripts/train_{grade_classifier,binary_sulfide,talc_segmentation,grain_classifier}.py`, `requirements-dev.txt`, `.gitignore`; appended distinct bullets to shared `ChangeLog.md`/`SMOKE_TESTS.md` | done |
| 2026-07-04T08:58:00+0300 | ch1-ch2-class-validation | Done: ran gx10 validation for class-folder ore classification across `Фото руд по сортам. ч1/ч2` with cross-class duplicate skipping and current sulfide+talc pipeline outputs; recorded metrics/docs. | `scripts/*classification*`, `scripts/*validation*`, `src/ore_classifier/resident_pipeline.py`, `outputs/evaluations/ch1_ch2_class_folder_talc_gate_20260704`, `docs/benchmarks/*`, `ChangeLog.md`, `docs/session-sync.md`, `docs/session-activity.md` | done |
| 2026-07-04T07:50:02+0300 | resilience-fixes | Done: fixed both HIGH bugs (resident accumulators → disk memmap; `grains_skipped_bad_bbox` init + recorded in `dataset_summary.json`). Added mandatory per-run degradation-record requirement (§4a) to plan 39. Both files py_compile. | `src/ore_classifier/resident_pipeline.py`, `scripts/build_grain_dataset.py`, `docs/plans/39_pipeline-resilience-and-recovery.md` | done |
| 2026-07-04T07:40:36+0300 | research+review | Done: external dataset/model scan, phase-aware intergrowth spec, review of parallel-session changes, session-coordination requirement, resilience plan (39). No code files edited — docs only. | `AGENTS.md`, `docs/session-activity.md`, `docs/specs/phase-aware-intergrowth-features.md`, `docs/notes/2026-07-04-external-datasets-models-sulfide-intergrowth.md`, `docs/plans/39_pipeline-resilience-and-recovery.md` (all created/edited) | done |

<!--
Template row (copy, fill, set status):
| <ISO ts> | <short label> | <what you are doing> | <files/dirs, plan/spec numbers> | active |
-->
