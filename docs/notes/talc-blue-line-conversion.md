# Talc Blue-Line Conversion

Date: 2026-07-03

This note records the v2 talc annotation conversion path for
`Области оталькования`.

## Purpose

The official talc annotations are not binary masks. They are microscope photos
with blue hand-drawn boundary strokes. Some strokes are open, and some marked
regions overlap bright sulfide grains. The v2 converter produces conservative
candidate talc masks plus QA artifacts for manual review.

## Code

- `src/ore_classifier/talc_blue_line_converter.py`: conversion core and review-mask helpers.
- `scripts/convert_talc_blue_lines.py`: CLI.
- `apps/talc_review_streamlit.py`: Streamlit QA app.
- `src/ore_classifier/sam2_region_assist.py`: optional SAM2 point/box assist, active only when `torch` and `sam2` are installed.
- `tests/test_talc_blue_line_converter.py`: focused synthetic unit tests.

## Command

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

Optional external sulfide masks can be supplied by image stem:

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --sulfide-mask-dir path/to/binary_sulfide_masks
```

## Outputs

Each sample directory contains:

- `raw_blue_stroke.png`
- `closed_blue_stroke.png`
- `filled_talc_region.png`
- `candidate_talc_mask.png`
- `sulfide_mask.png`
- `sulfide_overlap_mask.png`
- `ignore_mask.png`
- `final_talc_mask.png`
- `qa_overlay.png`
- `conversion_summary.json`

The current v2 full run is:

```text
outputs/talc_blue_line_conversion/manifest.json
sample_count: 42
candidate_ok: 31
needs_manual_review: 9
sulfide_overlap_review_required: 2
```

Samples requiring review:

```text
2550376-1 5x
2550377-1 5x
2550381-2 10x
2550382-1 10x
DSCN3042
DSCN3056
DSCN3057
DSCN4714
DSCN4715
DSCN4755
DSCN5180
```

## Review UI

```bash
streamlit run apps/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

The app displays original blue annotation lines next to the QA overlay by
default. Editing is mask-first: canvas defaults to `Current mask`, and the
original blue-line image remains available as a reference background.

The edit area uses a stateful `Editor` segmented control so the selected tool
survives Streamlit reruns after applying edits. Each editor shows local
`Current talc px`, `Current ignore px`, and `Unsaved edits` counters.

Canvas tools:

- `Pen` and `Eraser` apply stroke-width edits to the current mask.
- `Polygon` and `Box` apply filled areas; their line width is not the edited geometry.
- `Move/resize` exposes the Fabric transform mode for drawn canvas objects, including rectangle handles.

The `Geometry` editor uses the local component in
`apps/components/mask_shape_editor/index.html`. It supports polygon vertex
dragging, point insertion/deletion, and box corner/edge dragging before applying
the filled area as a mask. `Polygon table` and `Rectangle form` are kept as
exact-coordinate fallbacks.

The `SAM2` editor is optional. It now has explicit model/device controls and a
`Load/check SAM2` button before running point or box prompts. The model cache
should stay in the normal Hugging Face cache, not under the project tree.

Reviewed outputs are saved under each sample directory:

```text
reviewed/
  reviewed_talc_mask.png
  reviewed_ignore_mask.png
  reviewed_overlay.png
  review_patch.json
  review_summary.json
```

`Reload base masks` resets the current session masks and clears unsaved edit
history/canvas objects.

## Current Boundary

Convex-hull fallback is disabled by default because it overfills open strokes.
Use `--fallback-hull` only for exploratory QA, not final training masks without
manual review.
