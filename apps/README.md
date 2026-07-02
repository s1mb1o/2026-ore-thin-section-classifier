# Apps

Streamlit QA and visualization tools live here.

Implemented talc review app:

```bash
streamlit run apps/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

The talc app displays original blue annotation lines next to the current QA
overlay, but editing is mask-first. Canvas defaults to the current mask,
pen/eraser are stroke-width tools, polygon/box are filled-area tools, the
`Geometry` editor supports polygon vertex drag/add/delete plus box corner/edge
drag, and the SAM2 editor has optional model/device load-check controls. The
stateful `Editor` control survives reruns after apply actions and each edit
mode shows local current-mask counters. Reviewed talc/ignore masks are saved
under each sample's `reviewed/` directory.

Planned binary sulfide QA app:

```bash
streamlit run apps/sulfide_qa_streamlit.py -- \
  --pseudo-label-dir outputs/official_pseudo_labels \
  --output-dir outputs/binary_sulfide_qa
```

The QA app should save JSON patches and derived masks, not overwrite source pseudo-labels.
