# Plan 36: Source Disagreement Map For Ore Pipeline UI

Date: 2026-07-03

## Scope

This plan is for the v2 OM-only ore pipeline UI and pipeline artifacts. The
goal is to replace a weak standalone confidence heatmap story with a stronger
judge-facing "source disagreement map": a visual layer that shows where
different segmentation sources agree, where they conflict, and where pixels
should be excluded or reviewed.

Primary targets:

- `src/ore_classifier/source_fusion.py`
- `src/ore_classifier/review_queue.py`
- `scripts/run_ore_pipeline.py`
- `apps/ore_pipeline_web.py`
- run artifacts under `outputs/ore_pipeline_ui/runs/<run_id>/`

This is not a plan to claim pseudo labels as geological ground truth. The map
must explicitly frame every input as a source of evidence with known limits.

## Desired Jury Story

Instead of saying only "the model is confident here", the UI should show:

- model probability and final mask;
- teacher or Petroscope/LumenStone-style source mask when available;
- brightness/morphology heuristic baseline;
- optional TTA/ensemble instability;
- artifact, glare, ignore, and tile-border risk.

The visible story:

- green: active sources agree;
- yellow: partial agreement, for example 2 of 3;
- red: source conflict or instability;
- gray: excluded/ignored pixel.

This makes weak supervision honest and demonstrable: we can show where the
pipeline trusts the result, where it asks for review, and where it refuses to
pretend that a pseudo mask is ground truth.

## Source Model

### Required MVP Sources

1. `model`
   - Binary sulfide mask from the selected pipeline backend.
   - Probability/confidence map if available.
   - If only a hard mask exists, probability-derived uncertainty is marked as
     unavailable rather than fabricated.

2. `heuristic_baseline`
   - Sulfide mask from `heuristic_segmentation/` or the lightweight runtime
     heuristic path.
   - Used as an independent visual/morphology signal, not as ground truth.

3. `artifact_or_ignore`
   - User artifact mask, analyzed-area exclusion, black borders, blue annotation
     strokes, and any tile-border risk mask.
   - These pixels override the color map to gray when excluded from analysis.

### Optional Sources

4. `teacher`
   - Petroscope/LumenStone/other teacher mask only when a same-image or
     same-stem prepared mask is available.
   - The UI label must be `teacher / pseudo-label source`, not `ground truth`.

5. `tta_or_ensemble`
   - Instability from flip/scale TTA or B2/B1/B0/ResUNet ensemble disagreement.
   - Optional because it increases runtime. It can be enabled for selected demo
     samples and batch evidence, not necessarily every smoke run.

6. `tile_risk`
   - Tile seam/halo uncertainty from tiled inference.
   - Marked as a risk source when a pixel lies near a tile border or has high
     blend disagreement.

## Output Artifacts

Each run with disagreement enabled should write:

```text
uncertainty/
  source_manifest.json
  source_votes.json
  source_vote_count.png
  agreement_class_map.png
  disagreement_score.png
  disagreement_overlay.jpg
  review_candidates.csv
  review_candidates.json
```

Recommended meanings:

- `source_manifest.json`: active sources, missing sources, weights, thresholds,
  source artifact paths, and warnings.
- `source_votes.json`: compact per-source summary and pairwise agreement/IoU.
- `source_vote_count.png`: positive source count per pixel.
- `agreement_class_map.png`: encoded UI categories:
  - `0`: excluded/ignore;
  - `1`: full agreement;
  - `2`: partial agreement;
  - `3`: conflict/instability.
- `disagreement_score.png`: continuous score for review ranking.
- `disagreement_overlay.jpg`: RGB preview over source image.
- `review_candidates.*`: ranked crops from `review_queue`.

The existing plain model confidence/probability map should remain available,
but it becomes one source in this richer evidence stack.

## Color And Legend Contract

UI colors:

- green: all active non-ignore sources agree with the final decision;
- yellow: partial agreement, for example 2 of 3 active sources;
- red: conflict, instability, or strong source disagreement;
- gray: ignored/excluded area.

The legend must show active source count. Example:

```text
Источники: model, heuristic, teacher
Зеленый: 3/3 согласны
Желтый: 2/3 согласны
Красный: конфликт или нестабильность
Серый: исключено из анализа
```

When only two sources are active, the legend must change accordingly and avoid
claiming `2 of 3`.

## Implementation Plan

### Phase 1: Artifact Contract And Synthetic Core

1. Add a small focused module, preferably `src/ore_classifier/disagreement_map.py`,
   instead of overloading `source_fusion.py` with UI-specific categories.
2. Reuse existing `MaskSource`, `fuse_source_masks`, and
   `source_agreement_summary` for numeric source aggregation.
3. Add a categorical agreement builder:
   - input: named binary masks, optional probabilities, optional instability
     maps, optional valid/exclude mask;
   - output: vote count, agreement class map, disagreement score, summary.
4. Define deterministic category rules:
   - excluded pixels always become gray;
   - all active sources equal final mask -> green;
   - one source disagrees or a mid-score probability margin exists -> yellow;
   - multiple disagreement, TTA instability, artifact risk inside analyzed area,
     or tile-risk conflict -> red.
5. Add synthetic unit tests:
   - 3/3 agreement maps to green;
   - 2/3 agreement maps to yellow;
   - 1/3 or instability maps to red;
   - ignore mask overrides every category to gray;
   - missing optional source changes the denominator in summaries.

### Phase 2: Pipeline Artifact Generation

1. Extend `scripts/run_ore_pipeline.py` with an opt-in flag:
   - `--write-disagreement-map`;
   - optional `--teacher-mask`;
   - optional `--heuristic-mask`;
   - optional `--tta-disagreement` later.
2. For the heuristic backend in `apps/ore_pipeline_web.py`, generate the map
   from:
   - runtime heuristic sulfide mask;
   - model/source mask if available from a loaded ML run;
   - artifact/analyzed-area masks.
3. For the ML backend, generate from:
   - model sulfide probability/mask;
   - heuristic mask produced on the same resized analysis image;
   - optional teacher mask when available;
   - artifact/analyzed-area masks.
4. Store all uncertainty artifacts under immutable run directories.
5. Add warnings to `source_manifest.json` when a requested source is missing,
   resized, or incompatible.

### Phase 3: UI Layer

1. Add a new result viewer primary layer:
   - Russian: `сомнения`;
   - English: `disagreement`;
   - disabled until `agreement_class_map.png` exists.
2. Add side-by-side comparison support:
   - original vs disagreement;
   - final segmentation vs disagreement.
3. Add a compact legend next to the layer controls:
   - active source names;
   - source denominator;
   - color meanings;
   - warning if teacher is absent.
4. Add source toggles only if cheap:
   - `model`;
   - `teacher`;
   - `heuristic`;
   - `artifact/tile risk`.
5. Add pixel/region hover readout in a later pass:
   - active source votes at cursor;
   - final class;
   - disagreement category;
   - whether pixel is excluded.

### Phase 4: Review Queue Integration

1. Feed `disagreement_score.png` into `review_queue.build_review_queue`.
2. Add a `Review candidates` rail or modal in the result panel:
   - thumbnail crop;
   - bbox;
   - score;
   - reason, e.g. `source conflict`, `near threshold`, `tile border risk`.
3. Clicking a candidate should pan/zoom the viewer to that crop.
4. Export candidates to:
   - `review_candidates.csv`;
   - `review_candidates.json`;
   - future evidence bundle.

### Phase 5: TTA And Ensemble Instability

1. Add optional TTA for selected runs:
   - horizontal flip;
   - vertical flip;
   - small scale or crop shift if time allows.
2. Compute instability as probability variance or mask disagreement after
   inverse transform.
3. Keep TTA off by default in smoke/heuristic mode.
4. Add a visible runtime warning when TTA is not used:
   - `TTA instability source unavailable for this run`.
5. Later, allow ensemble disagreement from B2/B1/B0/ResUNet if checkpoints are
   available and runtime budget allows.

### Phase 6: Reporting And Evidence Bundle

1. Add disagreement section to PDF/report artifacts:
   - active sources;
   - conflict fraction;
   - ignored fraction;
   - top review candidates;
   - caveat: teacher/pseudo labels are not ground truth.
2. Include uncertainty artifacts in the future one-click evidence bundle:
   - maps;
   - overlays;
   - source manifest;
   - candidate crops.
3. Link the disagreement map from model/data/run cards when available.

## UI Copy

Russian result-panel copy:

```text
Карта сомнений показывает не уверенность одной модели, а согласие нескольких
источников. Зеленые зоны согласованы, желтые требуют внимания, красные спорные,
серые исключены из анализа. Teacher/pseudo-label источник используется только
как слабая разметка, не как ground truth.
```

Short tooltip:

```text
Сомнения = конфликт источников: модель, эвристика, teacher/псевдомаска,
TTA/ансамбль и зоны исключения.
```

## Acceptance Criteria

- A run can produce a disagreement map without a teacher source; the UI clearly
  shows the reduced active-source denominator.
- If a teacher/Petroscope/LumenStone mask is provided, it is labelled as a weak
  source, not ground truth.
- The UI displays `сомнения / disagreement` as a first-class result layer.
- The color legend is visible and adapts to active sources.
- Excluded pixels always render gray and are not counted as conflicts.
- `review_candidates.csv/json` ranks red/yellow regions using the reusable
  review queue.
- PDF/report/evidence outputs include source manifest and caveats.
- Synthetic tests cover agreement, partial agreement, conflict, and ignore
  cases.

## Risks And Guardrails

- Source alignment risk: teacher or heuristic masks may not match the current
  analysis-scale image. Use explicit size checks and record any resize.
- Runtime risk: TTA/ensemble can be too slow for demo. Keep it optional.
- Interpretation risk: green can be read as ground truth. Label it as source
  agreement, not truth.
- Color risk: final segmentation already uses green/red/blue. The disagreement
  layer needs its own legend and should not be shown as final-class colors.
- Data leakage risk: teacher masks from training/pseudo-label sources should be
  excluded from official evaluation claims unless provenance is explicit.

## First Practical Slice

Implement the smallest demo-worthy version:

1. Add `disagreement_map.py` with categorical source-agreement rules and tests.
2. Generate maps from `model mask + heuristic mask + artifact/analyzed mask`.
3. Add `сомнения` layer and legend to `apps/ore_pipeline_web.py`.
4. Export `review_candidates.csv/json` from `review_queue`.
5. Add teacher/TTA support only after the baseline source-disagreement layer is
   visible and tested.
