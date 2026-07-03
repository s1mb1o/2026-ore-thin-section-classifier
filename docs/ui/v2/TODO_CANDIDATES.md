# TODO Candidates: v2 Ore Pipeline UI

Date: 2026-07-03

Scope: candidate backlog for `apps/ore_pipeline_web.py` and closely related v2
browser tooling. This is not a commitment to implement every item. Pick only
items that improve the official OM-only demo, measured results, or review
workflow before widening the app surface.

Current UI baseline already includes upload, preprocessing, runtime
augmentation, metadata editing, immutable runs, batch runs, history, settings,
result layers, mask editing, CSV/PDF export, RU/EN localization, light/dark
theme support, Docker launch path, and a separate browser talc-mask review app.

## Latest Live UI Review

Reviewed on 2026-07-03 with the local browser apps:

- Ore pipeline UI: `apps/ore_pipeline_web.py`, heuristic backend, live pages
  `/workspace`, `/history`, `/batch`, and `/settings`.
- Talc review UI: `apps/talc_review_web.py`, full
  `outputs/talc_blue_line_conversion` workspace.
- Desktop viewport and `390 x 844` mobile smoke.

Confirmed already working:

- No global horizontal page overflow on the checked desktop/mobile viewports.
- History is now a full-width page with all/single/batch modes, load, remove,
  thumbnails, and metric columns.
- Loaded runs show result layers, opacity, contour mode, edit action, decision
  rationale, CSV export, and PDF export.
- Batch page exists with persisted gallery workflow.
- Settings persist language, theme, preprocessing defaults, and session metadata
  defaults.
- Talc review web app has sample filters, status tags, Brush/Fill/Rectangle/
  Polygon/SAM2 toolbar, zoom controls, direct talc-mask editing, save/save-next,
  sulfide protection, theme support, and no global mobile overflow.

Observed gaps to reflect in the backlog:

- The ore result layer chip row can still visibly truncate at normal desktop
  width when the left workflow sidebar is present; the screenshot showed a
  partial `с<-->` artifact near the layer controls.
- Backend/checkpoint/device readiness is not exposed in the UI.
- Demo sample loading is still manual through upload/history.
- Rule/calibration provenance is not visible beside a loaded result.
- There is no one-click evidence bundle ZIP.
- Review candidates/source-disagreement artifacts are not surfaced.
- Talc review queue labels are confusing after review: cards can show both the
  original conversion status such as `Needs manual review` and the final
  `Reviewed` state, while the header says `11 need review · 42 reviewed`.

## Selection Rules

- Prefer items that make the judged path clearer: image -> sulfide mask -> talc
  fraction -> ore class -> report.
- Keep SEM, XRD, defect/product-dashboard features out unless explicitly
  promoted.
- Treat weak labels and heuristic outputs as review evidence, not expert ground
  truth.
- Every implemented UI item should have a smoke-test expectation in
  `SMOKE_TESTS.md` and focused regression coverage when practical.

## P0 Candidates

| ID | Candidate | Why it matters | First implementation check |
|---|---|---|---|
| UI-P0-01 | Demo sample launcher | Speeds jury demo by loading known official examples, panorama crops, and the current strongest B2/B1/heuristic outputs without file hunting. | Add a small `Demo samples` selector that resolves repo-local paths and starts from existing immutable run artifacts when available. |
| UI-P0-02 | Backend readiness panel | Prevents silent demo failure when ML deps, checkpoint path, CUDA, or output writes are unavailable. | Show backend mode, checkpoint, device, model-load status, output root writability, and last error on `/workspace` and `/settings`. |
| UI-P0-03 | Rule/calibration artifact selector | The CLI supports `--rule-config-json`, but the UI should expose which deterministic thresholds are being used. | Add settings/run control for an `ore_rule_calibration.json` path and display applied thresholds in result metadata. |
| UI-P0-04 | Feature-classifier result lane | Current feature classifier is stronger than deterministic thresholds for image-level F1; UI should not hide that comparison. | Add optional "decision method" display: deterministic rule vs calibrated feature classifier, with clear experimental/provenance wording. |
| UI-P0-05 | Result warning strip | Near-threshold, zero-sulfide, low-analyzed-area, and talc-margin warnings need to be visible immediately. | Render `ore_summary.json` warnings and margins as a compact warning strip above metrics. |
| UI-P0-06 | Talc mask input/attachment | Auto talc is a candidate; reviewed masks should be selectable when stronger talc claims are needed. | Allow optional talc mask upload or same-stem reviewed-mask lookup before `Start`; record provenance in run metadata. |
| UI-P0-07 | One-click artifact bundle | Reviewers need the exact run evidence without browsing nested folders. | Add per-run `Download bundle` ZIP with input, masks, overlays, metrics, PDF, CSV, metadata, and rule config. |
| UI-P0-08 | Real panorama compliance run card | The official task mentions large panoramas; UI needs visible timing/memory proof for a panorama/crop run. | Show tile count, elapsed time, peak memory if available, tile size/stride, and whether target runtime was met. |
| UI-P0-09 | Result viewer toolbar fit | The loaded result UI can still clip the primary layer selector at normal desktop width, producing a visible partial control artifact. | Make primary/compare layer controls wrap cleanly or use a scrollable chip row with a visible affordance; add a 1280px regression check for no clipped chip text. |

## P1 Candidates

| ID | Candidate | Why it matters | First implementation check |
|---|---|---|---|
| UI-P1-01 | Batch retry/cancel controls | Batch currently runs sequentially; failed or slow items need practical recovery. | Keep existing batch-level cancel, then add per-item Retry/Cancel and Stop remaining without corrupting completed child runs. |
| UI-P1-02 | Batch summary dashboard | Judges and teammates need quick class/fraction distributions across a folder. | Add aggregate charts/table from `reports/batch_results.csv` with class counts, warning counts, and export links. |
| UI-P1-03 | History filters | The history table will become noisy during demo rehearsal. | Filter by filename, class, backend, warning status, date, and batch id. |
| UI-P1-04 | Parent-vs-derived comparison | `Fix and Restart` creates derived runs; users need to see what changed. | Add delta masks, changed-pixel counts, metric deltas, and parent/child links. |
| UI-P1-05 | Review queue integration | Existing `review_queue` library can rank high-impact uncertain regions; UI should surface them. | Add "Review candidates" rail based on uncertainty, margins, and source disagreement artifacts; see `plans/36_ore-pipeline-source-disagreement-map.md`. |
| UI-P1-06 | Source-fusion/disagreement overlay | Competing heuristic/model outputs are valuable trust evidence. | Add optional layer for disagreement maps from `source_fusion.py` when artifacts exist; see `plans/36_ore-pipeline-source-disagreement-map.md`. |
| UI-P1-07 | Editable crop bookmarks | Large panoramas need fast navigation to relevant problem areas. | Let users save named crop bookmarks with current zoom/pan, warning context, and export thumbnails. |
| UI-P1-08 | SAM2/scribble assist in `Fix me` | Talc review has richer edit tools than ore final-mask editing. | Add optional SAM2 or scribble-assisted region creation to the ore pipeline editor without making it required. |
| UI-P1-09 | PDF/report hardening | The current report exists; it should better explain method limits and provenance. | Add applied backend/checkpoint/rule config, analyzed-area definition, weak-label caveats, and warning summary to PDF. |
| UI-P1-10 | Clean-launch command copy | Demo setup should be copy-pasteable from the UI. | Settings/health panel shows current launch command, backend, port, and checkpoint path. |
| UI-P1-11 | Talc review queue semantics | After all samples are reviewed, showing `Needs manual review` beside `Reviewed` is confusing. | Split original conversion status from review state; filters should support `Unreviewed`, `Reviewed`, `Needs re-check`, and `Original conversion warning`. |
| UI-P1-12 | Talc review launcher/status bridge | Talc review is separate, but ore UI decisions depend on accepted talc masks. | Settings or talc input control shows reviewed-mask workspace status and a copyable `apps/talc_review_web.py` launch command. |

## P2 Candidates

| ID | Candidate | Why it matters | First implementation check |
|---|---|---|---|
| UI-P2-01 | Kiosk/presentation mode | Useful for a polished live demo, but not needed for the core result. | Hide dev controls and show only upload, run, result, and export actions. |
| UI-P2-02 | Run/model cards linked from UI | Provenance cards already exist in docs; UI can link them at the result level. | Add links to model card, dataset card, and run fact sheet when paths are available. |
| UI-P2-03 | Training-export page | Manual edits and accepted masks should become training-ready patches. | Export reviewed masks/metadata as a manifest for future training without changing source runs. |
| UI-P2-04 | Talc review launcher link | The talc review app is separate; the ore UI can help users jump there. | Add a Settings link showing the recommended `apps/talc_review_web.py` command and workspace path. |
| UI-P2-05 | Accessibility polish | Helpful for reliability, but secondary to judged pipeline proof. | Keyboard traversal for toolbar controls, visible focus rings, and aria labels for canvas tool state. |
| UI-P2-06 | Internationalized report templates | UI is RU/EN; reports can use richer localized wording later. | Add language-specific report text blocks without changing stored run metadata. |
| UI-P2-07 | Talc review Russian mode | The talc review app is currently English-first while the main ore UI defaults to Russian. | Add RU/EN language selector or at least Russian labels for demo-facing toolbar, filters, save state, and warnings. |

## Not Candidates For This File

- Broad SEM/XRD/product QC dashboard work.
- New model architectures without a matching metric/evaluation plan.
- UI-only decoration that does not improve demo reliability, evidence review, or
  result trust.
- A third talc-review UI; extend `apps/talc_review_web.py` or link to it.
