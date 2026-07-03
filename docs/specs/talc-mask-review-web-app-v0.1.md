# Talc Mask Review Web App v0.1

Date: 2026-07-03

Status: Implemented in `apps/talc_review_web.py`.

## Purpose

Replace the unreliable Streamlit mask-editing experience with a narrow local
browser application for talc mask QA. This app is talc-only because the source
annotations are hand-drawn MS Paint pen marks over talc regions. Those marks are
draft annotations, not masks: many regions are open, rough, and not reliable
closed shapes.

The app matches each annotated image from `Области оталькования` to its original
image in the parent `Оталькованные руды` folder by the same filename, derives an
initial talc mask from the pen annotations, and lets the user correct that talc
mask directly. It is not the training loop, final inference engine, sulfide QA
tool, or old broad OM/SEM/XRD platform.

## Proposed Product Decision

Build a lightweight local web app:

- Python backend with `http.server.ThreadingHTTPServer` or equivalent minimal
  stdlib server.
- Generated HTML/CSS plus vanilla JavaScript.
- Browser `canvas` for image display, mask overlay, polygon/rectangle editing,
  brush/eraser editing, pan/zoom, and SAM2 prompts.
- File-based outputs compatible with the current converter and training
  manifests.

This satisfies the optional web-interface/dashboard requirement without
depending on Streamlit/Dash/Gradio templates. It keeps the computation in the
existing Python stack: OpenCV, scikit-image, torch, SAM2, and project code under
`src/ore_classifier/`.

## Input Pairing and Preparation

The review app works directly from the official blue-line annotation folder and
the original talcose-ore image folder.

Source folders:

- annotated images:
  `dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования`
- original images:
  `dataset/Фото руд по сортам. ч1/Оталькованные руды`

The app pairs files by exact filename. For example:

```text
dataset/.../Оталькованные руды/Области оталькования/DSCN3042.JPG
dataset/.../Оталькованные руды/DSCN3042.JPG
```

The annotated image is used to detect draft talc regions from hand-drawn pen
marks. The original image is used as the clean review background and final
overlay source.

Primary launch input:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --original-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

`--original-dir` defaults to the parent of `--annotated-dir`, so the shorter
command is also valid for the official folder layout:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

On startup, the app checks whether the workspace contains a current manifest and
per-sample working masks. If the workspace is missing or `--reconvert` is
requested, it runs the same conversion logic as:

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

The conversion copies or references each blue-line annotation image, extracts
blue strokes, closes/fills candidate regions, subtracts or marks sulfide
overlap, and writes per-sample mask artifacts plus
`outputs/talc_blue_line_conversion/manifest.json`.

The new app must additionally record the matched clean original image path for
each sample. If an annotated file has no original file with the same filename,
the sample remains visible in the queue with `missing_original` status and no
editing canvas until the match is fixed.

Optional display-only evidence masks can be supplied by image stem:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --original-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --sulfide-mask-dir path/to/binary_sulfide_masks \
  --silicate-mask-dir path/to/silicate_support_masks
```

Prepared-workspace mode is still supported for reproducibility/debugging:

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

If neither a raw input folder nor a valid conversion manifest is available, the
app should show a clear setup screen with the exact direct-input command.

## Scope

### In Scope for v0.1

- Accept the official `Области оталькования` folder directly through
  `--annotated-dir`.
- Match each annotated image to the clean original image by the same filename in
  `--original-dir`.
- Create or reuse a talc conversion/review workspace under
  `outputs/talc_blue_line_conversion`.
- Auto-save the autodetected mask as the current working talc mask when a sample
  is first opened.
- Load the generated talc-only review manifest from that workspace.
- Show a review queue with status, reason, source image name, and existing
  candidate statistics.
- Display the clean original photo, the annotated MS Paint draft, the
  autodetected talc region, the current talc mask, and optional display-only
  evidence layers such as sulfide overlap or silicate support.
- Edit the talc mask directly, not the pen strokes.
- Protect talc-mask edits from sulfide overlap by default and provide an
  explicit manual action to subtract sulfide pixels from the current mask.
- Save reviewed masks under each sample's `reviewed/` directory.
- Preserve an auditable edit patch with vector geometry and/or mask deltas.
- Keep SAM2 optional and lazy-loaded.
- Provide exact-coordinate fallback forms behind an advanced panel.
- Run locally without internet after dependencies and model cache are present.

### Out of Scope

- Sulfide QA or ordinary/fine intergrowth QA.
- Old broad OM/SEM/XRD upload portal.
- XRD, SEM registration, defect/product platform UI.
- Editing ignore masks as a primary user task.
- Multi-user authentication or public deployment.
- Live model training from the UI.
- Downloading model weights into the repository.
- Treating non-expert edits as geological ground truth without review status.

## Users

- Reviewer: corrects talc masks from draft MS Paint annotations.
- ML engineer: uses reviewed talc masks and patches for training data.
- Presenter: demonstrates original annotation, converted mask, manual
  correction, SAM2 assist, and final reviewed overlay.

## Primary Workflow

1. Launch the app with the official `Области оталькования` annotated folder.
2. App matches each annotated file to the original file with the same filename.
3. App creates or reuses `outputs/talc_blue_line_conversion`.
4. App opens the generated review queue.
5. Select a sample from the queue.
6. On first open, app saves the autodetected talc mask as the current working
   talc mask.
7. Inspect the clean original, annotated MS Paint draft, autodetected mask, and
   current talc mask.
8. Edit the current talc mask with brush, eraser, filled polygon, filled
   rectangle, or SAM2 assist.
9. Use undo when an edit is wrong.
10. Press `Save` to persist the current sample.
11. Press `Save and next` to persist the current sample and move to the next
    sample in the queue.

## UX Requirements

### Layout

The first screen is the working review interface, not a landing page.

- Left column: sample queue with filters.
- Center: canvas viewer and editing controls.
- Right column: sample details, layer toggles, current metrics, save status.
- Bottom or collapsible side panel: edit history and advanced exact-coordinate
  tools.
- Theme selector: `System`, `Light`, and `Dark`, persisted in browser storage.

The reviewer must always see that the edited object is the `Talc mask`.

### Queue Filters

- All samples.
- Needs review.
- Sulfide overlap.
- Candidate OK.
- Reviewed.
- Missing original.
- Search by sample id/stem.

### Viewer Layers

Base image modes:

- Original photo.
- MS Paint annotated photo.
- QA overlay.
- Current mask view.

Overlay toggles:

- Talc mask.
- Autodetected talc region.
- Sulfide overlap, display-only.
- Raw blue strokes.
- Silicate support, display-only when available.
- SAM2 preview mask, when available.

Layer opacity must be adjustable without rerunning Python.

The right panel must include a default-on `Protect sulfides while drawing`
control. When enabled, additive tools clip newly applied talc pixels against
the loaded sulfide mask without silently changing pre-existing overlap. A
separate `Subtract sulfides from mask` command removes any existing talc pixels
on the sulfide mask and autosaves the working mask. `Current on sulfide px`
shows whether any overlap still exists.

### Canvas Interaction

Required controls:

- Pan and zoom by mouse/trackpad, plus reset view.
- Brush add.
- Eraser/remove.
- Filled polygon.
- Filled rectangle.
- SAM2 assist.
- Undo.
- Redo.
- Clear draft geometry.
- Save.
- Save and next.
- Apply draft to `Talc mask`.

Brush width applies only to brush and eraser. Polygon, rectangle, and SAM2
operate on filled regions.

Edits update the current working talc mask. The app should auto-save this
working mask after each applied edit so leaving and reopening a sample does not
lose work. `Save` marks the current working mask as reviewed and writes the
review patch; `Save and next` does the same and navigates to the next queue row.

### Polygon Tool

Polygon editing must support:

- Add point by clicking empty canvas.
- Insert point by clicking an existing polygon edge.
- Drag existing points.
- Delete point by right-clicking it.
- Close polygon automatically when at least three points exist.
- Preview filled region before applying.
- Apply as add/remove/replace operation to the talc mask.

### Rectangle Tool

Rectangle editing must support:

- Draw by drag.
- Drag corners.
- Drag edges.
- Move whole rectangle.
- Preview filled region before applying.
- Apply as add/remove/replace operation to the talc mask.

### SAM2 Tool

SAM2 must be a canvas tool, not a separate coordinate-only form.

Prompt modes:

- Positive point.
- Rectangle.
- Polygon/rectangle refinement if practical in v0.1; otherwise keep as v0.2.

SAM2 behavior:

- Show model/device/load status.
- Load lazily only after explicit reviewer action.
- Use default Hugging Face/model cache, not project-local model downloads.
- Display returned mask as a preview layer.
- Reviewer must explicitly apply the SAM2 preview to the talc mask.
- If SAM2 is unavailable, show a local dependency message and keep manual tools
  usable.

## Data Model

### Sample Record

Each sample shown by the app is resolved from `manifest.json` and per-sample
`conversion_summary.json`.

Required fields:

```json
{
  "sample_id": "DSCN3042",
  "sample_dir": "outputs/talc_blue_line_conversion/DSCN3042",
  "annotated_image": "dataset/.../Области оталькования/DSCN3042.JPG",
  "original_image": "dataset/.../Оталькованные руды/DSCN3042.JPG",
  "raw_blue_stroke": "raw_blue_stroke.png",
  "autodetected_talc_mask": "final_talc_mask.png",
  "current_talc_mask": "current_talc_mask.png",
  "sulfide_overlap_mask": "sulfide_overlap_mask.png",
  "qa_overlay": "qa_overlay.png",
  "status": "needs_manual_review",
  "match_status": "matched",
  "review_status": "unreviewed"
}
```

Optional fields:

- `silicate_support_mask`
- `silicate_supported_talc_mask`
- `silicate_unsupported_talc_mask`
- `talc_positive_core_mask`
- `silicate_hard_negative_mask`
- `reviewed_talc_mask`
- `review_patch`

### Mask Semantics

- `autodetected_talc_mask`: initial mask produced from MS Paint pen annotations.
- `current_talc_mask`: editable working talc mask. Created automatically from
  `autodetected_talc_mask` when a sample is first opened and updated after
  applied edits.
- `reviewed_talc_mask`: final saved mask for training/review export.
- `sulfide_overlap_mask`, `silicate_support_mask`, and related masks:
  display-only evidence layers for review; they are not edited by this app.
- `not_talc`: implicit background outside the reviewed talc mask. Any uncertainty
  handling for training manifests is downstream of this app.

## Patch Format

Every save writes both raster masks and an auditable patch.

```json
{
  "schema_version": "talc-mask-review-patch-v0.1",
  "sample_id": "DSCN3042",
  "reviewer": "",
  "review_status": "reviewed",
  "source": {
    "conversion_summary": "conversion_summary.json",
    "annotated_image": "dataset/.../Области оталькования/DSCN3042.JPG",
    "original_image": "dataset/.../Оталькованные руды/DSCN3042.JPG",
    "autodetected_talc_mask": "final_talc_mask.png",
    "base_working_talc_mask": "current_talc_mask.png"
  },
  "edits": [
    {
      "edit_id": "edit_0001",
      "timestamp_utc": "2026-07-03T00:00:00Z",
      "tool": "filled_polygon",
      "operation": "add",
      "geometry": {
        "type": "polygon_xy",
        "points": [[10, 10], [100, 20], [80, 90]]
      },
      "notes": ""
    }
  ],
  "outputs": {
    "reviewed_talc_mask": "reviewed/reviewed_talc_mask.png",
    "reviewed_overlay": "reviewed/reviewed_overlay.png",
    "review_summary": "reviewed/review_summary.json"
  }
}
```

Operations:

- `add`
- `remove`
- `replace`

Tools:

- `brush`
- `eraser`
- `filled_polygon`
- `filled_rectangle`
- `sam2_point`
- `sam2_rectangle`
- `uploaded_mask`
- `advanced_coordinates`

## Backend API

The app can be implemented with manual routes. API responses are JSON unless
serving artifact bytes.

### Pages

- `GET /`: main app shell.
- `GET /samples/{sample_id}`: app shell preselected to one sample.

### Artifacts

- `GET /artifacts/{relative_path}`: serve only allowlisted files under the
  configured conversion/review directories.

### JSON Endpoints

- `GET /api/conversion/status`
- `POST /api/conversion/reconvert`
- `GET /api/manifest`
- `GET /api/samples/{sample_id}`
- `GET /api/samples/{sample_id}/mask-state`
- `POST /api/samples/{sample_id}/autosave-working-mask`
- `POST /api/samples/{sample_id}/draft`
- `POST /api/samples/{sample_id}/sam2-assist`
- `POST /api/samples/{sample_id}/save-review`
- `POST /api/samples/{sample_id}/reload-base`
- `POST /api/samples/{sample_id}/set-review-status`
- `GET /api/health`

### Save Review Request

```json
{
  "review_status": "reviewed",
  "reviewer": "",
  "notes": "",
  "edits": [],
  "current_talc_mask_png_base64": "",
  "navigate": "stay"
}
```

The backend validates dimensions, writes masks, writes the patch JSON, and
returns updated metrics. `navigate` may be `stay` for `Save` or `next` for
`Save and next`.

## File Layout Proposal

Implemented files:

```text
apps/talc_review_web.py
tests/test_talc_review_web.py
```

The application remains narrow and local: generated HTML/CSS/JavaScript live in
the Python app, while tests cover pairing, first-open autosave, reviewed saves,
theme controls, and HTTP endpoints without requiring a browser.

## Launch Command

Primary direct-input command:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

Prepared-workspace command:

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

No fixed default port is assigned in this spec. Before implementation chooses a
default port, check `/Users/ashmelev/Projects/10_admin/mbp2023/PORTS_USED.md`
and update it if a new persistent default is reserved.

## Validation

### Unit Tests

- Patch schema validates required fields.
- Direct-input startup creates a conversion workspace when no manifest exists.
- Direct-input startup matches annotated/original files by exact filename and
  reports `missing_original` samples.
- Prepared-workspace startup loads an existing manifest without reconverting.
- Opening a sample for the first time copies `autodetected_talc_mask` to
  `current_talc_mask`.
- Polygon, rectangle, brush, eraser, and SAM2 mask previews convert to expected
  raster masks on synthetic inputs.
- Save-review rejects wrong-size masks.
- Artifact serving rejects paths outside the configured conversion directory.

### Browser/UX Smoke

Use Playwright or equivalent browser automation.

Required checks:

- App opens and loads the manifest.
- App launched with `--annotated-dir` shows conversion progress or a completed
  conversion status before opening the queue.
- Canvas renders a nonblank source image.
- Layer toggles update without a Python rerun.
- Zoom/pan works and does not change mask geometry.
- Polygon points can be added, dragged, deleted, and applied.
- Rectangle corners/edges can be dragged and applied.
- Brush/eraser modify only the talc mask.
- Undo restores prior mask state.
- Applying an edit updates/autosaves `current_talc_mask`.
- With sulfide protection enabled, brush/polygon/rectangle/SAM2 additions over
  sulfide pixels are clipped and the status line reports newly protected pixels
  without silently cleaning older overlap.
- `Subtract sulfides from mask` removes existing overlap, autosaves, and leaves
  `Current on sulfide px` at zero.
- `Save` writes reviewed masks and patch JSON.
- `Save and next` writes reviewed masks and patch JSON, then selects the next
  sample in the queue.
- SAM2 unavailable state does not block manual editing.

### Manual Acceptance

- A reviewer can correct one `needs_manual_review` talc sample in under two
  minutes after launch.
- The app makes it visually obvious that the reviewer is editing the talc mask.
- Original blue lines remain visible as reference but are never the edited
  object.
- Saved reviewed overlay matches the current canvas state.

## Migration Plan

1. Keep `apps/talc_review_streamlit.py` available as fallback.
2. Implement direct-input startup that creates or reuses the same conversion
   output directory.
3. Keep prepared-workspace mode for debugging/reproducibility.
4. Verify on at least one `candidate_ok`, one `needs_manual_review`, and one
   `sulfide_overlap_review_required` sample.
5. Update `COMMANDS.md` with the new launch command.
6. Mark Streamlit as deprecated only after the new app passes unit and browser
   smoke tests.

## Review Questions

1. Should the save output remain exactly under each sample's `reviewed/`
   directory, or should reviewed masks also be collected into a global training
   manifest immediately?
2. Should SAM2 polygon refinement be required for v0.1, or is point/rectangle
   enough for the first reliable replacement?
3. Should the app keep a reviewer name field, or is anonymous local review
   sufficient?
4. Should Streamlit remain as a documented fallback after the new app is
   accepted, or should it be removed to avoid two QA paths?

## Proposed v0.1 Acceptance Criteria

- The app has no Streamlit runtime dependency.
- The app can start directly from the official `Области оталькования` folder.
- The app matches each annotated file to the clean original by identical
  filename.
- The app can review all `42` talc samples after creating or reusing the
  conversion workspace.
- The app edits the current talc mask, not blue annotation strokes.
- Brush draws talc; eraser removes talc.
- Polygon, rectangle, and SAM2 are direct filled-area tools for drawing talc.
- Sulfide protection is enabled by default; additive tools cannot add new talc
  pixels on the sulfide mask unless the reviewer disables protection.
- Manual sulfide subtraction is available and autosaves the current working
  mask.
- Undo is available for editing mistakes.
- `Save` and `Save and next` are both available.
- SAM2 is available as a canvas tool and optional dependency.
- Reviewed outputs are raster masks plus a machine-readable patch JSON.
- UI work is local and file-based, so model training and inference remain
  runnable from CLI without the app.
