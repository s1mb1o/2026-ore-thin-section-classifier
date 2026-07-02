# Apps

Streamlit QA and visualization tools live here.

Implemented talc review app:

```bash
streamlit run apps/talc_review_streamlit.py -- \
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

Implemented binary sulfide QA app:

```bash
streamlit run apps/sulfide_qa_streamlit.py -- \
  --runs-dir outputs/inference_demo \
  --review-dir outputs/sulfide_qa_reviews
```

The current app reviews precomputed pipeline outputs: sulfide overlay,
confidence heatmap, binary mask, component overlay, and JSON summaries. It saves
review verdict JSON files without overwriting source masks. Pixel-level binary
mask editing remains a follow-up; training/inference stay in CLI scripts.
