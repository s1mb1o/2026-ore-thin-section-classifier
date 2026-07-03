# Ore Pipeline v2 UI/UX QA Review

Date: 2026-07-03

Reviewer stance: senior UI/UX QA with mineralogy domain focus. Scope was the live local v2 ore pipeline UI at `http://127.0.0.1:63589/`, with completed run and batch artifacts already present.

## Tested Surfaces

- Workspace empty state and selected-run state.
- Result viewer with final segmentation.
- Edit & Recalculate dialog, artefacts and final-segmentation layers.
- History page all-runs and batches modes.
- Batch detail page.
- Settings page.
- English/Russian localization switch.
- 390 x 844 mobile viewport smoke.

## Implementation Status

Implemented on 2026-07-03 in `apps/ore_pipeline_web.py`:

- P1 fixes: mobile overflow removal, full-width non-workspace pages, scroll reset after run load, editor default layer from current context, explicit metric/stat denominators, and Russian localization cleanup for the identified static labels.
- P2/P3 fixes: neutral disabled-button styling, adaptive main-canvas height, overlay opacity plus contour-only mode, shorter editor tabs, `Панорама` -> `Перемещение`, no-wrap numeric stats, and a decision-rationale text line with margins/warnings.
- Verification: `python3 -m py_compile apps/ore_pipeline_web.py`; `python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v`; in-app browser checks for `/workspace`, `/history`, and temporary `390 x 844` viewport with no horizontal overflow.

## Priority Issues

### P1. Mobile layout has horizontal overflow and inaccessible navigation

Evidence: at `390 x 844`, `documentElement.scrollWidth > innerWidth`; the header nav starts at `x=176` with width `372`, and primary layer controls are `556 px` wide. The user sees a horizontal scrollbar and clipped top navigation.

Impact: field/demo use on laptop side panels, tablets, or narrow windows breaks. The reviewer may not be able to reach `History`, `Settings`, or result layer controls without horizontal scrolling.

Proposed fix:
- Make header controls a responsive grid: title full width, language/theme row, nav row with wrapping.
- Convert wide segmented layer controls to horizontally scrollable chips with visible scroll affordance, or to a compact two-row grid at small widths.
- Add regression checks for `documentElement.scrollWidth <= innerWidth` at `390 x 844` and `768 x 1024`.

### P1. History and Batch pages are constrained by the left workflow sidebar

Evidence: on `/history` at about `1047 px` viewport width, the history table wrapper is `610 px` wide while the table is `1040-1162 px`; action columns are off-screen. Batch history mode has the same problem. Settings and Batch detail also inherit the workflow sidebar and become cramped.

Impact: History is an analytical review surface, not an image-entry workflow. Keeping the upload/preprocessing sidebar visible reduces the table to half width, hides actions, and makes batch comparison difficult.

Proposed fix:
- For non-workspace pages (`/history`, `/batch`, `/settings`), hide/collapse the left workflow sidebar or convert it to a slim drawer.
- Keep the sidebar only on `/workspace`, where upload/run controls are the primary task.
- Make History actions sticky/right-pinned if a table still scrolls horizontally.

### P1. Loading a run can preserve stale scroll position

Evidence: after loading a run from a batch, the app navigated to `/workspace`, but the primary layer toolbar had `y=-22`; the screenshot showed the top of the result controls clipped above the viewport.

Impact: the user sees a partially hidden result viewer and may miss layer controls, class toggles, and the current state after `Load`.

Proposed fix:
- On `loadRun()` and page changes to Workspace from History/Batch, call `window.scrollTo({top: 0, left: 0})` after rendering.
- Alternatively, keep the viewer toolbar sticky within the workspace panel.

### P1. Edit & Recalculate opens on Artefacts even when user is reviewing Final

Evidence: while the main viewer was on `финал`, pressing `Исправить` opened the editor with `артефакты` active.

Impact: this breaks user intent. If the geologist notices a wrong fine/ordinary/talc region in the final map, they are dropped into an unrelated artifact-exclusion layer and may paint the wrong mask.

Proposed fix:
- Default editor layer from current main viewer layer: `artefacts -> artifact`, `sulfide -> sulfide`, `final -> final`.
- For `original`, `augmented`, or `preprocessed`, default to `artifact` before a run and `final` after a completed run.
- Add a small active-layer subtitle in the editor header: `Editing: final segmentation`.

### P1. Editor class percentages use a different denominator from classification output

Evidence: result text for the loaded run says fine intergrowth predominance is `95.5%`, while the editor final-layer stats show fine as `61.24%`. The latter is whole analyzed/image area, while the decision uses fine/ordinary within sulfides.

Impact: mineralogy reviewers will read this as a contradiction. For ore classification, ordinary/fine intergrowth proportions must be clear as sulfide-normalized values.

Proposed fix:
- In final segmentation stats, show both:
  - area share of analyzed image, and
  - share of sulfides for ordinary/fine.
- Label denominators explicitly, for example `тонкие срастания: 1 361 408 px · 61.24% image · 95.53% sulfides`.
- In the result metrics table, add a short denominator note under the table.

### P1. Russian UI still contains hard-coded English labels

Evidence in Russian mode: `Augmentation`, `Edit`, `Batch`, `batches`, `sidecar прибора`, `Augmentation settings`, `Color and tone`, and related augmentation popup labels are hard-coded or intentionally English. English mode static labels were mostly clean; Cyrillic detected there came from a filename.

Impact: language quality is inconsistent in the default Russian UI and undermines demo polish.

Proposed fix:
- Localize all static Russian-mode labels, except deliberate product terms. Suggested Russian labels:
  - `Augmentation` -> `Аугментация`
  - `Edit` -> `Настроить...`
  - `Batch` -> `Пакет` or consistently keep `Batch` everywhere as a product term
  - `batches` -> `пакеты`
  - `instrument sidecar` -> `служебный файл прибора`
- Add a static test that scans default Russian HTML for known English UI strings outside an allowlist.

### P2. Disabled primary/danger buttons look actionable

Evidence: disabled `Запустить Batch`, `Старт`, and `Исправить и перезапустить` retain strong teal/red fills with only reduced opacity. In dark mode these still read as active CTA buttons.

Impact: users can mistake disabled controls for available actions, especially during demos or long-running analysis.

Proposed fix:
- Use neutral disabled styling for all buttons: muted text, neutral background, no accent fill.
- Keep semantic danger/primary colors only when enabled.
- Add tooltip/status explanation for disabled high-value actions such as `Start`, `Run Batch`, and `Fix and Restart`.

### P2. Main result canvas wastes vertical review space

Evidence: loaded final segmentation is centered in a `~899 px` high canvas, leaving large black bands above/below the actual image.

Impact: for OM thin-section review, local mineral texture, sulfide morphology, and talc/artefact boundaries should be visible without unnecessary empty space.

Proposed fix:
- Make canvas height adaptive to active image aspect ratio with min/max constraints.
- Add visible `Fit width`, `Fit image`, and current zoom percentage controls in the main viewer.
- For result view, place metrics beside the image when width allows instead of far below.

### P2. Final segmentation overlay lacks opacity and denominator context near the image

Evidence: the final map uses requested green/red/blue overlays, but red/green can obscure texture or blend into greenish gangue/talc matrix. The class controls are checkboxes without opacity/confidence.

Impact: mineralogy QA often needs to compare segmentation boundaries against polished-section texture. Fully saturated overlays reduce confidence in boundary edits and artifact diagnosis.

Proposed fix:
- Add overlay opacity slider, default around `55-65%`.
- Keep class visibility checkboxes but add a small legend with pixel/% values beside the viewer.
- Add optional boundary-only mode for final classes.

### P2. Editor layer tabs wrap and truncate class names

Evidence: in the right panel, `сульфиды/не сульфиды` wraps to three lines and `финальная сегментация` is visually cramped/truncated.

Impact: the layer selector consumes attention and looks unstable. It is especially problematic for high-stakes edits where the selected mask determines what recalculates.

Proposed fix:
- Use shorter tab labels: `Артефакты`, `Сульфиды`, `Финал`.
- Put the full explanation below the tabs.
- Consider vertical radio-style layer list on narrow right panels.

### P2. Pan tool is translated as `Панорама`

Evidence: the editor toolbar uses `Панорама` for the pan/move-view tool.

Impact: in this app, `панорама` already means a panorama image/scaling mode. Using it for panning creates ambiguity.

Proposed fix:
- Rename to `Перемещение` or `Рука`.
- Use an icon-only hand button with tooltip if icons are added.

### P2. Editor statistics wrap labels and large values

Evidence: editor stats show values like `2 223 000 px`, where `px` can wrap to the next line. A later live smoke on 2026-07-03 also showed Russian class labels such as `сульфиды`, `не-сульфиды`, `обычные срастания`, and `тонкие срастания` wrapping into very narrow letter-sized columns in the right statistics panel.

Impact: scanning pixel counts and percentages becomes harder during mask correction.

Proposed fix:
- Use a stable stats grid with a fixed/minmax label column and fixed numeric columns.
- Apply `white-space: nowrap` to numeric values and prevent class labels from collapsing below a readable minimum width.
- Format large pixel counts with a compact unit option if space is tight, for example `2.22M px`.

### P3. Result decision lacks threshold/margin explanation

Evidence: result text says the ore is classified as hard-to-process because fine intergrowth predominates, but no rule threshold, margin, or uncertainty is visible next to the decision.

Impact: domain reviewers need to know whether a case is robust or near a decision boundary, especially for ordinary vs fine intergrowth and talcose classification.

Proposed fix:
- Add a `Decision rationale` row/card: talc threshold, fine/ordinary share, margin, and warnings from `summary.warnings`.
- Use explicit labels: `fine share among sulfides`, `talc share of analyzed area`.

## Recommended Fix Order

1. Fix non-workspace layout width and mobile horizontal overflow.
2. Fix run-load scroll reset and editor default layer.
3. Fix denominator labeling in final stats and result metrics.
4. Finish Russian localization for all static controls.
5. Improve disabled button styling and editor tab labels.
6. Add overlay opacity/boundary mode and decision rationale.

## Follow-Up Live Review

Date: 2026-07-03

Scope:

- Rechecked `apps/ore_pipeline_web.py` through the local browser app on
  `/workspace`, `/history`, `/batch`, and `/settings`.
- Rechecked `apps/talc_review_web.py` on the full
  `outputs/talc_blue_line_conversion` workspace.
- Used desktop viewport plus `390 x 844` mobile smoke.

Confirmed improvements:

- The old global mobile horizontal overflow is gone on the checked pages.
- History is now full-width rather than constrained by the workflow sidebar.
- A loaded run exposes layer controls, opacity, contour-only mode, edit action,
  decision rationale, CSV export, and PDF export.
- Batch and Settings pages are available through slug routes.
- Talc review web app is usable at desktop and mobile widths with direct
  Brush/Fill/Rectangle/Polygon/SAM2 controls, sample filters, save/save-next,
  sulfide protection, and status tags.

Remaining findings now reflected in `docs/ui/v2/TODO_CANDIDATES.md`:

- Result viewer layer controls can still truncate visibly at normal desktop
  width when the left sidebar is present; one screenshot showed a partial
  `с<-->` artifact near the primary layer chips.
- Edit & Recalculate statistics still need layout hardening: class labels can
  collapse into very narrow wrapped columns in the right panel.
- Backend/checkpoint/device readiness and launch command are not shown in the
  UI.
- Rule/calibration provenance is not visible beside the loaded result.
- Demo sample loading still relies on manual upload/history selection.
- No one-click evidence bundle ZIP is exposed.
- Review candidates and source-disagreement overlays from the reusable
  libraries are not surfaced in the UI.
- Talc review status semantics need cleanup: cards can show both an original
  conversion warning such as `Needs manual review` and final `Reviewed`, while
  the header reports `11 need review · 42 reviewed`.
