# Playwright UI QA — Ore Pipeline & Talc Review (2026-07-04)

Senior-QA pass driving both browser apps headless with Playwright (Chromium 1.58).
Fresh instances on dedicated ports with clean workspaces:

- Ore pipeline: `python3 apps/ore_pipeline_web.py --host 127.0.0.1 --port 9310 --backend heuristic --workspace-dir <scratch>/ore_ws`
- Talc review: `python3 apps/talc_review_web.py --host 127.0.0.1 --port 9311 --conversion-dir outputs/talc_blue_line_conversion`

## Coverage exercised

**Ore pipeline** — all 7 routes load (`/workspace`, `/batch`, `/history`, `/history_series`,
`/status`, `/api`, `/settings`); end-to-end upload → Start → run → overlay/metrics/rationale →
history card; CSV/PDF/ZIP download endpoints (all HTTP 200, correct content types); "View files"
popup (sortable table, sizes, dims, ZIP footer); "Fix me" artefact editor opens with canvas + tools;
language RU↔EN toggle; theme System/Dark toggle; API sandbox executes (JSON + HTTP 200); Series and
Settings pages render and expose their actions. **Console/network clean on every page.**

**Talc review** — 42 samples list with filters/status; canvas renders image + positive-bag/talc
overlays at full 2272×1704; toolbar Brush/Fill/Similar/Rectangle/Polygon/SAM2/Undo/Zoom/Fit + Save/
Save&Next/Next all present and interactive; brush stroke registers, autosaves working mask, updates
live stats. **Console/network clean.** (English-only UI — acceptable for an internal review tool.)

## Issues

### 1. [Medium] CSV/PDF "Save" export lacks `Content-Disposition` — PDF opens inline / no filename
- **Where**: `apps/ore_pipeline_web.py` — `GET /api/runs/{id}/metrics.csv` (line ~4872) and
  `GET /api/runs/{id}/report.pdf` (line ~4876) call `send_file(...)` **without** `download_name`,
  so no `Content-Disposition` header is sent. Frontend `<a id="csvLink">`/`<a id="pdfLink">`
  (`apps/static/ore_pipeline_ui.html` lines 542–543) have no `download` attribute either.
- **Effect**: Clicking "Сохранить CSV" / "Сохранить PDF-отчет" navigates the SPA in the same tab to
  the raw file (PDF renders inline, replacing the app; no meaningful filename). Contrast: the ZIP
  endpoint (line ~4884) correctly passes `download_name=` and downloads as an attachment.
- **Verified**: `curl`/urllib — `metrics.csv` → 200 `text/csv`, **no** disposition; `report.pdf` →
  200 `application/pdf`, **no** disposition; `artifacts.zip` → 200 with
  `Content-Disposition: attachment; filename*=...`.
- **Fix**: pass `download_name=f"{run_id}_metrics.csv"` / `f"{run_id}_report.pdf"` to `send_file`,
  mirroring the ZIP endpoint.
- **Status**: FIXED.

### 2. [Low] Run-progress stage labels leak raw English into the localized (default Russian) UI
- **Where**: `stageLabel()` in `apps/static/ore_pipeline_ui.html` (line ~4660) maps only a subset of
  stages; unknown stages fall back to the raw backend string (`return stage`). Backend
  `_set_progress(...)` emits these stage strings that are **not** matched:
  `"preparing immutable run artifacts"`, `"ordinary/fine intergrowth and talc analysis"`,
  `"running ML tiled inference"`, `"collecting ML outputs"`, `"building display layers"`.
- **Effect**: During every run (including the default Russian UI) the progress caption shows raw
  English stage text, e.g. `building display layers · 100% · осталось 0 с`.
- **Fix**: add keyword branches to `stageLabel()` and matching `stage*` i18n keys in both RU and EN
  dictionaries.
- **Status**: FIXED.

### 3. [Low, observed] Language switch after a completed run doesn't re-localize the last progress caption
- Switching RU↔EN after a run finishes leaves the final progress caption in the previous language
  (it isn't re-rendered). Edge case; visibility reduced by fix #2. **Not fixed** — documented only.

## Test harness

Playwright driver scripts in the session scratchpad (`lib.mjs`, `ore_nav.mjs`, `ore_e2e.mjs`,
`ore_interact.mjs`, `ore_api_series.mjs`, `talc.mjs`). Screenshots under `scratchpad/shots/`.

**Note**: the talc brush test autosaved a stroke into sample `2550374-2 10х` working masks
(`current_*`); restored exactly from the intact `reviewed/` masks (positive_bag 257,898 / union
324,022 / talc_node 71,890). Reviewed deliverables were never touched.
