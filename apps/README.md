# Apps

Local QA and visualization tools live here.

Implemented ore pipeline UI:

```bash
python3 apps/ore_pipeline_web.py \
  --host 127.0.0.1 \
  --port 0
```

The ore pipeline app accepts PNG, JPEG, TIFF, and RAW-extension uploads, builds
display previews for large images, optionally applies deterministic
geometry-preserving runtime augmentation before preprocessing, including
color/tone shifts, acquisition noise, and grinding/polishing artifacts such as
scratches, polishing haze, and pits/dust specks. It applies illumination
normalization, denoising, contrast correction, and panorama scaling presets,
edits curated metadata through `Edit Metadata...`, creates immutable run
artifacts, shows original/augmented/preprocessed/sulfide/final/side-by-side
views, exports metrics CSV/PDF reports, supports `Fix me` mask edits, and
creates a new derived run for every `Fix and Restart`. When curated metadata
contains a calibrated positive `microns_per_pixel` or `pixel_size_um`, the
result metrics table and CSV export include pixel area, physical area, and scale
provenance; without calibrated scale metadata, physical area fields stay empty.

The same app includes the v2 `Batch` page at `/batch`: add multiple images,
edit per-image metadata, share one preprocessing/augmentation setup across the
group, run images sequentially into separate immutable runs, monitor each card's
progress, and load any child run back into the normal result viewer.

## Deprecated Streamlit apps

The legacy Streamlit tools were superseded by the plain web apps
(`talc_review_web.py`, `ore_pipeline_web.py`) and now live under
`apps/deprecated/streamlit/` together with their custom
`mask_shape_editor` component. They still run, but are kept only as
fallbacks and are no longer actively maintained.

Legacy talc review app:

```bash
streamlit run apps/deprecated/streamlit/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

The talc app displays original blue annotation lines next to the current QA
overlay, but editing is mask-first. The main `Workspace` is `Review canvas`:
`Brush` and `Erase` are stroke-width tools, `Filled polygon` and `Filled box`
open editable filled areas with polygon vertex add/delete/drag and box
corner/edge drag, and `SAM2 assist` uses draggable point/box prompts plus
optional model/device load-check controls. `Advanced` keeps exact coordinate
fallbacks out of the normal review path. Reviewed talc/ignore masks are saved
under each sample's `reviewed/` directory.

When the converter is run with `--silicate-mask-dir`, the app starts from the
supported talc candidate and shows unsupported candidate pixels as ignore. Use
the generated `talc_positive_core_mask.png` and `silicate_hard_negative_mask.png`
for training exports, not the raw silicon/silicate mask as talc.

Legacy binary sulfide QA app:

```bash
streamlit run apps/deprecated/streamlit/sulfide_qa_streamlit.py -- \
  --runs-dir outputs/inference_demo \
  --review-dir outputs/sulfide_qa_reviews
```

The current app reviews precomputed pipeline outputs: sulfide overlay,
confidence heatmap, binary mask, component overlay, and JSON summaries. It saves
review verdict JSON files without overwriting source masks. Pixel-level binary
mask editing remains a follow-up; training/inference stay in CLI scripts.
