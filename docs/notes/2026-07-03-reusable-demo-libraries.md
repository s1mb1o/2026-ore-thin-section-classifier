# Reusable Demo Libraries

Date: 2026-07-03

This note records the first implementation pass that turns the research mindstorm ideas into reusable code for the final hackathon demo.

## Implemented Shared Modules

- `src/ore_classifier/source_fusion.py`
  - Reusable source-vote fusion for heuristic/model/SAM/manual masks.
  - Produces weighted probability, fused mask, positive vote counts, disagreement map, and agreement summary.

- `src/ore_classifier/review_queue.py`
  - Active-review queue builder for crops ranked by `uncertainty * decision_impact * novelty`.
  - Produces bbox records and Russian expert-question prompts.

- `src/ore_classifier/curation.py`
  - Lightweight dataset curation helpers without FiftyOne/cleanlab dependencies.
  - Supports image feature vectors, uniqueness scores, near-duplicate pairs, hardness maps, and segmentation label-issue masks from predicted probabilities.

- `src/ore_classifier/component_reports.py`
  - Component-level report helpers inspired by OIA/MLA outputs.
  - Supports association contacts, sulfide component liberation proxies, talc/ordinary/fine decision margins, and expert-review flags.

- `src/ore_classifier/report_cards.py`
  - Markdown renderers for model cards, dataset cards, and run fact sheets.
  - Intended for final reproducibility/provenance artifacts.

- `src/ore_classifier/scribble_classifier.py`
  - Dependency-light scribble pixel classifier inspired by ilastik/Labkit/Weka/QuPath.
  - Extracts RGB/grayscale/multiscale local features and fits a nearest-centroid classifier from sparse foreground/background scribbles.

## Test Coverage

Synthetic unit tests were added for every module:

- `tests/test_source_fusion.py`
- `tests/test_review_queue.py`
- `tests/test_curation.py`
- `tests/test_component_reports.py`
- `tests/test_report_cards.py`
- `tests/test_scribble_classifier.py`

Current verification command:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Result on 2026-07-03: `31` tests passed.

## Demo Integration Ideas

Shortest useful final-demo path:

1. Run binary sulfide inference and heuristic segmentation.
2. Feed masks into `source_fusion.fuse_source_masks`.
3. Feed `disagreement` plus decision-impact maps into `review_queue.build_review_queue`.
4. Build component outputs with `component_analysis` and enrich with `component_reports`.
5. Export `model_card.md`, `dataset_card.md`, and `run_fact_sheet.md` through `report_cards`.

Optional quick interactive path:

1. Let a reviewer draw sparse foreground/background scribbles.
2. Fit `scribble_classifier.fit_scribble_pixel_classifier`.
3. Add the scribble classifier mask as another source in `source_fusion`.

## Limits

- These modules do not replace full model training.
- `curation.py` intentionally provides lightweight approximations, not full FiftyOne/cleanlab parity.
- `scribble_classifier.py` is a quick adaptation baseline; for final production it should be compared against RF/ExtraTrees if `scikit-learn` is allowed.
