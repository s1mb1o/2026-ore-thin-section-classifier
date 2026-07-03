# B2 Manual Review Pack

Date: 2026-07-03

## Scope

Prepared a manual review and feedback pack for the current default binary sulfide checkpoint:

- checkpoint: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt` locally, generated on zelda from `outputs/train_segformer_b2_zelda_20260703_overnight_safetensors/best.pt`;
- source split: `outputs/official_balanced_eval_split.json`;
- selected images: `3` ordinary, `3` fine/hard-to-process, `3` talcose;
- output directory: `outputs/manual_review/b2_balanced_review_pack/`;
- source subset copy: `outputs/manual_review/source_dataset_subset/`;
- generated review candidates: `8` uncertainty crops.

This is a human review artifact, not a statistical benchmark. It is intended to expose obvious mask errors, ordinary/fine rule disagreements, and talcose cases where talc integration is still missing.

## Generated Files

- `outputs/manual_review/b2_balanced_review_pack/review_manifest.csv` and `.json`: full run index, metrics, image paths, overlays, masks, and panels.
- `outputs/manual_review/b2_balanced_review_pack/feedback_template.csv`: spreadsheet-friendly feedback form.
- `outputs/manual_review/b2_balanced_review_pack/review_candidates.csv`: uncertainty crop index with Russian review prompts.
- `outputs/manual_review/b2_balanced_review_pack/runs/*/review_panel.jpg`: 2x2 visual panel with source, sulfide overlay, probability heatmap, and ordinary/fine overlay.
- `outputs/manual_review/b2_balanced_review_pack/runs/*/candidate_crops/*.jpg`: focused source/overlay/confidence crops for high-uncertainty areas.

## Streamlit Review

Run from the v2 root:

```bash
streamlit run apps/sulfide_qa_streamlit.py -- \
  --runs-dir outputs/manual_review/b2_balanced_review_pack/runs \
  --review-dir outputs/manual_review/b2_balanced_review_pack/reviews
```

The Streamlit QA app now displays optional `review_panel`, `source_preview`, and `confidence_heatmap` paths when they are present in `pipeline_summary.json`.

## Summary Table

| Source label | Review id | Predicted class | Sulfide fraction | Ordinary sulfide | Fine sulfide |
| --- | --- | --- | ---: | ---: | ---: |
| ordinary_intergrowth | `01_ordinary_intergrowth_2539589-2` | hard_to_process_ore | 0.4333 | 0.1410 | 0.8519 |
| ordinary_intergrowth | `02_ordinary_intergrowth_DSCN0474` | hard_to_process_ore | 0.2652 | 0.1808 | 0.8067 |
| ordinary_intergrowth | `03_ordinary_intergrowth_DSCN8635` | row_ore | 0.9013 | 0.9997 | 0.0000 |
| fine_intergrowth | `04_fine_intergrowth_2539444-1` | hard_to_process_ore | 0.2186 | 0.2935 | 0.6977 |
| fine_intergrowth | `05_fine_intergrowth_90` | hard_to_process_ore | 0.1514 | 0.4561 | 0.5118 |
| fine_intergrowth | `06_fine_intergrowth_DSCN8767` | row_ore | 0.1752 | 0.6322 | 0.3229 |
| talcose | `07_talcose_2550374-2_10х` | hard_to_process_ore | 0.2232 | 0.1509 | 0.8418 |
| talcose | `08_talcose_-3` | row_ore | 0.0392 | 0.6797 | 0.2810 |
| talcose | `09_talcose_DSCN7483` | row_ore | 0.0609 | 0.9522 | 0.0290 |

## Review Guidance

Use the pack to answer these questions:

1. Is the binary sulfide mask visually acceptable?
2. Are there obvious false sulfide or missed sulfide areas?
3. Does the ordinary/fine overlay match the geological intuition for the labelled class?
4. For talcose source labels, mark `talc_issue` when the result is wrong because the current B2 pack does not use accepted talc masks yet.
5. For high-uncertainty crops, mark whether the crop is a useful correction example for active learning.

Suggested statuses for `feedback_template.csv` or Streamlit QA:

- `accepted`
- `needs_mask_fix`
- `uncertain`
- `exclude_artifact`
- `bad_input`

Suggested error types:

- `missed_sulfide`
- `false_sulfide`
- `bad_boundary`
- `wrong_ordinary_fine`
- `talc_issue`
- `artifact`

## Caveats

- The local macOS environment could not load the B2 checkpoint because its installed `transformers` package uses a different SegFormer module namespace. The pack was generated on zelda, where the training environment uses `transformers 5.12.1` and CUDA.
- The generated `image_path` values point to the local copied subset under `outputs/manual_review/source_dataset_subset/`, so the review pack remains self-contained without duplicating the full official dataset.
- Talc masks are not yet wired into `run_ore_pipeline.py`; talcose samples are included to collect visible feedback and to highlight this remaining integration gap.
