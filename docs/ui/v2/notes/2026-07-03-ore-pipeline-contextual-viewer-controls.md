# Ore Pipeline Contextual Viewer Controls

Date: 2026-07-03

The v2 ore pipeline result viewer uses layer-specific visibility controls.

## Contract

- Primary layers stay ordered as `original`, `augmented`, `preprocessed`, `sulfide`, `final`.
- Side-by-side comparison stays ordered as `none`, `augmented`, `preprocessed`, `sulfide`, `final`.
- `original`, `augmented`, and `preprocessed` do not show segmentation class visibility controls unless the side-by-side layer is `sulfide` or `final`.
- If either the primary layer or side-by-side layer is `sulfide`, show only:
  - sulfides;
  - non-sulfides;
  - artefacts.
- If either the primary layer or side-by-side layer is `final`, show only:
  - ordinary intergrowths;
  - fine intergrowths;
  - talc;
  - artefacts;
  - background.
- When both `sulfide` and `final` are visible through side-by-side, show both control groups.
- `show tiling`, `contours only`, and `opacity` are viewer-level controls and sit in one row below the layer selectors, not mixed with segmentation class controls.

## Rendering Semantics

- In `sulfide`, `non-sulfides` controls the base image visibility, `sulfides` controls the sulfide overlay, and `artefacts` controls the violet/magenta artefact overlay.
- In `final`, `background` controls the base image visibility, while ordinary, fine, talc, and artefacts control their overlays.
- The same contextual controls affect the primary view and the side-by-side right view because both are rendered through the same composite-layer path.
