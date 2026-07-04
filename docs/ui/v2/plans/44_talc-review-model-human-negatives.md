# Plan 44: Talc Review Model/Human QA And Hard Negatives

Date: 2026-07-04

## Goal

Upgrade `apps/talc_review_web.py` from a positive-mask editor into a QA station
for talc model errors and teammate disagreement, while preserving the existing
reviewed-mask contract used by training scripts.

## Scope

- Add editable `Not Talc` hard-negative class.
- Save `current_not_talc_mask.png` and `reviewed/reviewed_not_talc_mask.png`.
- Keep `current_talc_mask.png` and `reviewed/reviewed_talc_mask.png` as
  compatibility unions of `Positive bag | Talc`; `Not Talc` is not part of that
  union.
- Add optional trained model talc mask display and model-vs-current-human QA
  overlay: agreement, model only, human only, sulfide conflict.
- Add optional teammate human mask loading from extra review directories and a
  human agreement/disagreement overlay.
- Improve Similar with explicit positive/negative seeds and local texture
  features in addition to luma/color.

## Implementation Notes

- Server changes stay local to `TalcReviewStore`: add optional
  `--talc-model-mask-dir` and repeatable `--human-review-dir`.
- Model/human masks are read-only artifact URLs. If absent, the UI keeps working
  and reports that the QA layer is unavailable.
- `Not Talc` is editable by Brush, Fill, Rectangle, and Polygon through the
  existing edit-target radio pattern.
- Similar excludes sulfide pixels, existing talc pixels, `Not Talc` pixels, and
  pixels that are closer to negative seeds than positive seeds.
- Review patch JSON records class definitions, negative seed metadata, model QA
  settings, and hard-negative mask paths.

## Verification

- Focused unit tests for save/read payload paths and page contract.
- `python3 -m py_compile apps/talc_review_web.py`.
- `python3 -m unittest tests.test_talc_review_web -v`.
- `git diff --check` for touched docs/code.
