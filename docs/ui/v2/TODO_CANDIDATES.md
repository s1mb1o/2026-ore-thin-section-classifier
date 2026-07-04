# TODO Candidates: v2 Ore Pipeline UI

Date: 2026-07-04

Scope: candidate backlog for `apps/ore_pipeline_web.py` and closely related v2
browser tooling. This is not a commitment to implement every item. Pick only
items that improve the official OM-only demo, measured results, or review
workflow before widening the app surface.

Current UI baseline already includes upload, preprocessing, runtime
augmentation, metadata editing, immutable runs, Series, History, Settings,
Status, API docs/sandbox, runtime testing, result layers, mask editing, grain
table/outlines, CSV/PDF export, `View files`, `Download ZIP`, runtime
provenance, RU/EN localization, light/dark theme support, Docker/gx10 launch
path, and a separate browser talc-mask review app.

## Latest Current-State Review

Reviewed on 2026-07-04 from the current docs/code state and prior browser-app
QA:

- Ore pipeline UI: `apps/ore_pipeline_web.py`, heuristic backend, live pages
  `/workspace`, `/history`, `/batch`, `/settings`, `/status`, and `/api`.
- Talc review UI: `apps/talc_review_web.py`, full
  `outputs/talc_blue_line_conversion` workspace.
- Current state note: `docs/notes/2026-07-04-current-state-ideas-review.md`.

Confirmed already working:

- No global horizontal page overflow on the checked desktop/mobile viewports in
  the last live QA pass.
- History is a full-width page with all/single/Series modes, load, remove,
  thumbnails, metrics, and active job state overlay.
- Loaded runs show result layers, opacity, contour mode, edit action, decision
  rationale, runtime details, grain table/outlines, CSV export, PDF export,
  file browser, and ZIP export.
- Series page exists with persisted gallery workflow and sequential execution.
- Settings persist language, theme, preprocessing defaults, runtime backend,
  binary/talc checkpoint settings, talc source, and session metadata defaults.
- Status/API pages expose runtime readiness, logs, app backend/checkpoints, and
  API contracts.
- Talc review web app has sample filters, status tags, Brush/Fill/Rectangle/
  Polygon/SAM2 toolbar, zoom controls, direct talc-mask editing, save/save-next,
  sulfide protection, theme support, and no global mobile overflow.

Observed gaps to reflect in the backlog:

- The ore result layer chip row can still visibly truncate at normal desktop
  width when the left workflow sidebar is present; the screenshot showed a
  partial `с<-->` artifact near the layer controls.
- Workspace has no compact "demo-ready" health strip even though Status/Settings
  now expose backend/checkpoint readiness.
- Demo sample loading is still manual through upload/history.
- Rule/calibration provenance is not visible enough beside a loaded result, and
  UI launch still cannot select `--rule-config-json` like the CLI can.
- The new Path A ordinary/fine CNN result is not surfaced as a decision lane in
  the UI/report; the UI still mostly tells the segmentation/rule story.
- Talc model source/checkpoint is wired, but fraction accuracy is not yet good
  enough for a `+/-3 pp` claim; UI/report must avoid overclaiming.
- Review candidates/source-disagreement artifacts are not surfaced.
- Real panorama compliance evidence is not condensed into a visible run card.
- ML Series remains per-item execution-bound; resident inference exists but is
  not integrated into the browser Series path.
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
| UI-P0-01 | Demo sample launcher | The demo should load known row/fine/panorama/talc evidence without manual file hunting. | Add a small `Demo samples` selector that opens prepared immutable runs first and falls back to repo-local input paths when no run exists. |
| UI-P0-02 | Decision lane provenance | Path A now beats the old rule path for ordinary/fine, but it is 2-class and must not be mixed with full 3-class claims. | Add a result block for `decision method`: deterministic rule, feature-CV evidence, Path A ordinary/fine, and talc branch status, each with measured-scope wording. |
| UI-P0-03 | Rule/calibration artifact selector | The CLI supports `--rule-config-json`; the UI/report must show exactly which thresholds were used. | Add Settings/run control for an `ore_rule_calibration.json` path, pass it into starts, and display applied thresholds in runtime/result metadata. |
| UI-P0-04 | Talc fraction claim guard | Talc IoU is strong enough to show masks, but fraction MAE `8.551` pp makes `+/-3 pp` unsafe. | Show talc model source/checkpoint/threshold and a warning/caveat when fraction calibration has not passed the target. |
| UI-P0-05 | Source-disagreement layer | This is the strongest remaining judge-facing "research" feature: source agreement, not a plain confidence heatmap. | Implement the first slice from `plans/36`: model mask + heuristic mask + artifact/analyzed mask -> `agreement_class_map.png`, `disagreement_overlay.jpg`, UI layer `сомнения`. |
| UI-P0-06 | Review candidates from disagreement | Existing `review_queue` can turn red/yellow regions into actionable QA crops. | Export `review_candidates.csv/json` and add a compact result rail/modal with reason, bbox, score, and click-to-focus. |
| UI-P0-07 | Real panorama compliance run card | The official task mentions panoramas; UI needs visible timing/tile proof without relying on the public VM. | Implement `plans/40_ore-pipeline-panorama-compliance-card.md`: show input size, analysis size, tile count, elapsed time, backend/device, and whether the run was live or pre-generated. |
| UI-P0-08 | Result viewer toolbar fit | The loaded result UI can still clip the primary layer selector at normal desktop width, producing a visible partial control artifact. | Make primary/compare layer controls wrap cleanly or use a scrollable chip row with a visible affordance; add a 1280px regression check for no clipped chip text. |
| UI-P0-09 | Workspace demo health strip | Status page exists, but presenters need a compact readiness signal on the main screen. | Show backend, binary checkpoint, talc source/checkpoint, device, output writability, and last runtime-test result above Start. |

## P1 Candidates

| ID | Candidate | Why it matters | First implementation check |
|---|---|---|---|
| UI-P1-01 | Series retry/cancel controls | Series runs sequentially; failed or slow items need practical recovery. | Keep existing Series-level cancel, then add per-item Retry/Cancel and Stop remaining without corrupting completed child runs. |
| UI-P1-02 | Series resident path bridge | Resident inference removes checkpoint reload overhead, but browser Series still follows per-item pipeline execution. | Add a runtime option or backend path that uses resident batch inference when the selected model/device supports it. |
| UI-P1-03 | History filters | The history table will become noisy during demo rehearsal. | Filter by filename, class, backend, warning status, date, and batch id. |
| UI-P1-04 | Parent-vs-derived comparison | `Fix and Restart` creates derived runs; users need to see what changed. | Add delta masks, changed-pixel counts, metric deltas, and parent/child links. |
| UI-P1-05 | Batch summary dashboard | Judges and teammates need quick class/fraction distributions across a folder. | Add aggregate charts/table from `reports/batch_results.csv` with class counts, warning counts, and export links. |
| UI-P1-06 | Evidence ZIP hardening | ZIP exists; now it must be complete enough for final judging evidence. | Ensure runtime provenance, rule config, model references, disagreement artifacts, reports, masks, overlays, metrics, and warnings are included. |
| UI-P1-07 | Editable crop bookmarks | Large panoramas need fast navigation to relevant problem areas. | Let users save named crop bookmarks with current zoom/pan, warning context, and export thumbnails. |
| UI-P1-08 | SAM2/scribble assist in `Fix me` | Talc review has richer edit tools than ore final-mask editing. | Add optional SAM2 or scribble-assisted region creation to the ore pipeline editor without making it required. |
| UI-P1-09 | PDF/report hardening | The current report exists; it should better explain method limits and provenance. | Add applied backend/checkpoint/rule config, analyzed-area definition, weak-label caveats, and warning summary to PDF. |
| UI-P1-10 | Clean-launch command copy | Demo setup should be copy-pasteable from the UI. | Settings/health panel shows current launch command, backend, port, checkpoint path, talc source, and rule config path. |
| UI-P1-11 | Talc review queue semantics | After all samples are reviewed, showing `Needs manual review` beside `Reviewed` is confusing. | Split original conversion status from review state; filters should support `Unreviewed`, `Reviewed`, `Needs re-check`, and `Original conversion warning`. |
| UI-P1-12 | Talc review launcher/status bridge | Talc review is separate, but ore UI decisions depend on accepted talc masks. | Settings or talc input control shows reviewed-mask workspace status and a copyable `apps/talc_review_web.py` launch command. |
| UI-P1-13 | Grain-review bridge | Path B is implemented but only becomes useful with real human labels. | Add a link/status to `apps/grain_review_web.py` and show how many grain labels exist for the current run/dataset. |

## P2 Candidates

| ID | Candidate | Why it matters | First implementation check |
|---|---|---|---|
| UI-P2-01 | Kiosk/presentation mode | Useful for a polished live demo, but not needed for the core result. | Hide dev controls and show only upload, run, result, and export actions. |
| UI-P2-02 | Run/model cards linked from UI | Provenance cards already exist in docs; UI can link them at the result level. | Add links to model card, dataset card, and run fact sheet when paths are available. |
| UI-P2-03 | Training-export page | Manual edits and accepted masks should become training-ready patches. | Export reviewed masks/metadata as a manifest for future training without changing source runs. |
| UI-P2-04 | Talc review launcher link | Superseded by the stronger P1 launcher/status bridge unless only a simple link is needed. | Add a Settings link showing the recommended `apps/talc_review_web.py` command and workspace path. |
| UI-P2-05 | Accessibility polish | Helpful for reliability, but secondary to judged pipeline proof. | Keyboard traversal for toolbar controls, visible focus rings, and aria labels for canvas tool state. |
| UI-P2-06 | Internationalized report templates | UI is RU/EN; reports can use richer localized wording later. | Add language-specific report text blocks without changing stored run metadata. |
| UI-P2-07 | Talc review Russian mode | The talc review app is currently English-first while the main ore UI defaults to Russian. | Add RU/EN language selector or at least Russian labels for demo-facing toolbar, filters, save state, and warnings. |
| UI-P2-08 | Full MIL/CLAM experiment surface | Research-interesting, but Path A already gives a strong ordinary/fine branch. | Keep as an offline notebook/model experiment until it has measured lift over Path A or adds interpretability Path B cannot provide. |

## Implemented / Superseded Ideas

- `Download ZIP` and file browser are implemented; remaining work is bundle
  completeness, not adding a new endpoint.
- Backend readiness exists in Settings/Status; remaining work is a compact
  Workspace health strip.
- Result warning/rationale exists; remaining work is stronger talc-fraction and
  decision-lane caveats.
- Broad "train a grade classifier" is no longer just an idea: Path A exists and
  should be integrated carefully rather than duplicated.

## Not Candidates For This File

- Broad SEM/XRD/product QC dashboard work.
- New model architectures without a matching metric/evaluation plan.
- UI-only decoration that does not improve demo reliability, evidence review, or
  result trust.
- A third talc-review UI; extend `apps/talc_review_web.py` or link to it.
- Public Nornickel VM workloads or redeploys after the 2026-07-04 stop request.
