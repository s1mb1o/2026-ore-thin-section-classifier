# Weak-Supervision Sulfide Binary Model Plan

Date: 2026-07-03

## Purpose

Define a practical training path for the first-stage `sulfide / not_sulfide` model when no geologist is available for full expert mask correction.

This plan supports the official `Скажи мне, кто твой шлиф` pipeline:

```text
image
-> binary sulfide model
-> sulfide components and replacement features
-> ordinary_intergrowth / fine_intergrowth
-> separate talc detector in non-sulfide matrix
-> deterministic ore class rule
```

The plan deliberately does not treat pseudo-labels as ground truth. It uses LumenStone, Petroscope, simple image-processing baselines, and uncertainty masking to build a useful binary sulfide detector without overclaiming expert validation.

## Key Decision

Use a weak-supervision teacher-student path:

```text
LumenStone mineral masks -> supervised pretraining for sulfide / not_sulfide
Petroscope predictions on official images -> teacher pseudo-labels
brightness / morphology baseline -> independent pseudo-label source
agreement zones -> high-confidence training masks
disagreement zones -> ignore / uncertain, not hard labels
student model -> final binary sulfide detector
```

Manual review remains available, but it is not positioned as expert geological labeling. Without a geologist, manual edits should be limited to visually obvious errors: missed bright sulfide regions, included dark matrix, image borders, glare, scratches, dirt, or clearly invalid areas.

## Why Binary First

`ordinary_intergrowth` and `fine_intergrowth` are not pure pixel-color classes. They describe the morphology of a sulfide inclusion:

- ordinary: large, compact, mostly continuous sulfide;
- fine: sulfide fragmented or strongly penetrated/replaced by non-ore dark/gray phase.

Therefore the first model should answer a simpler and better-supervised question:

```text
where are sulfide pixels?
```

The intergrowth classifier then works on connected components, reconstructed footprints, dark-inclusion ratio, boundary complexity, fragmentation, and size.

Talc is a separate detector:

```text
not_sulfide matrix -> talc detector from official blue-line annotations
```

Do not map generic `silicate` labels to talc. Talc is a silicate mineral, but most silicates are not talc.

## Data Sources

### LumenStone

Use for supervised pretraining because it has real polished-section microscopy and pixel-level mineral masks.

Expected mapping:

```text
sulfide-positive:
  chalcopyrite, pyrite, pyrrhotite, pentlandite, bornite,
  galena, sphalerite, tennantite and other sulfide/sulfosalt labels

not_sulfide:
  background, silicate/gangue, resin, void, oxides, artifacts
```

Magnetite is an oxide, not a sulfide. For the binary sulfide model it should default to `not_sulfide`, but it can remain a useful dark/gray replacement-phase cue for later ordinary-vs-fine morphology.

### Petroscope

Use as a teacher and baseline, not as the final truth source.

Flow:

```text
official image -> Petroscope ResUNet -> mineral mask -> sulfide / not_sulfide pseudo-mask
```

If Petroscope and the LumenStone-pretrained student agree, the region is likely useful as high-confidence pseudo-label. If they disagree, mark it as `ignore` for training and surface it in debug/QA views.

### Brightness / Morphology Baseline

Use a simple non-neural baseline as a third independent weak label source:

- illumination normalization;
- bright mineral thresholding;
- morphology closing/opening;
- small-object filtering;
- artifact/border exclusion.

This baseline is especially useful for detecting gross teacher failures and for explaining the method to judges.

### Official Dataset

Use official class folders as image-level calibration data:

- `рядовые` / `Рядовые руды`;
- `тонкие` / `Труднообогатимые руды`;
- `оталькованные` / `Оталькованные руды`.

These labels should calibrate the downstream ore decision and component thresholds, not be treated as binary sulfide pixel masks.

Use `Области оталькования` only for talc supervision after converting blue-line annotations into masks with QA.

## Label Fusion

For each official image, produce:

```text
petroscope_sulfide_mask.png
lumenstone_student_mask.png
heuristic_sulfide_mask.png
agreement_mask.png
ignore_mask.png
pseudo_sulfide_mask.png
qa_overlay.png
```

Suggested fusion rules:

```text
if at least two strong sources agree on sulfide:
    label = sulfide
elif at least two strong sources agree on not_sulfide:
    label = not_sulfide
else:
    label = ignore
```

Optional confidence weights:

- high: Petroscope + student + heuristic agree;
- medium: Petroscope + student agree;
- low: only one source predicts sulfide;
- ignore: contradictory labels or artifact-like area.

Training loss should ignore `ignore` pixels.

## Model Choice

Start with a simple binary segmentation model:

1. **ResUNet / U-Net** for the first runnable P0:
   - easy to train;
   - easy to package;
   - enough for binary sulfide detection;
   - compatible with high-resolution tiled inference.

2. **SegFormer-B0/B1** as the likely quality upgrade:
   - better texture/context handling;
   - already used in the repository;
   - still practical on T4-class GPU.

3. **Mask2Former** only after the binary path is stable:
   - heavier runtime;
   - more setup risk;
   - better kept as optional experiment, not P0.

## Review App Scope Without a Geologist

Use a dedicated Streamlit app for Phase 3.

The Streamlit app should be framed as QA and weak-label cleanup, not expert mineralogical annotation. It is a review tool over precomputed masks, not the training or inference engine.

Proposed entry point:

```text
submissions/ore_classifier/apps/sulfide_qa_streamlit.py
```

Example command:

```bash
streamlit run submissions/ore_classifier/apps/sulfide_qa_streamlit.py -- \
  --pseudo-label-dir outputs/official_pseudo_labels \
  --output-dir outputs/binary_sulfide_qa
```

Required layers:

- original image;
- Petroscope pseudo-mask;
- student prediction;
- heuristic mask;
- agreement / disagreement overlay;
- ignored pixels;
- final pseudo-mask;
- later: ordinary/fine/talc final mask.

Allowed edit labels:

```text
sulfide
not_sulfide
exclude_artifact
uncertain
```

Avoid asking a non-geologist to mark `ordinary_intergrowth` vs `fine_intergrowth` manually except for obvious visual sanity checks. Those labels should be calibrated from official image-level classes and component statistics.

### Streamlit App Behavior

Keep the app simple and file-based:

1. Load the pseudo-label manifest and list images with counts of disagreement / ignore pixels.
2. Show the selected image with layer toggles:
   - original;
   - pseudo sulfide mask;
   - ignore mask;
   - agreement / disagreement overlay;
   - Petroscope-only, student-only, heuristic-only masks.
3. Prioritize review queue by disagreement area, low confidence, or large sulfide-component impact.
4. Allow region edits:
   - `sulfide`;
   - `not_sulfide`;
   - `uncertain`;
   - `exclude_artifact`.
5. Save edits immediately to a JSON patch; never overwrite the source pseudo-label files.
6. Apply patch to produce:
   - `corrected_sulfide_mask.png`;
   - `corrected_ignore_mask.png`;
   - `training_manifest.json`.

For drawing, prefer a Streamlit canvas component if available. If that dependency is not acceptable for the final environment, keep a fallback rectangle editor with coordinate inputs and small crop previews. Polygon/brush editing is useful but not required for P0.

Do not run model training from Streamlit. The app may emit the exact next training command, but GPU jobs should run as separate CLI jobs on `gx10` / `zelda`.

### Binary QA Patch Contract

Use a narrow patch schema instead of the generic phase-correction patch:

```json
{
  "schema_version": "binary-sulfide-qa-patch-v0.1",
  "image_id": "sample_001",
  "image_path": "dataset/Фото руд по сортам. ч2/рядовые/sample.jpg",
  "base_sulfide_mask": "official_pseudo_labels/sample_001/pseudo_sulfide_mask.png",
  "base_ignore_mask": "official_pseudo_labels/sample_001/ignore_mask.png",
  "edits": [
    {
      "edit_type": "binary_label_assignment",
      "target_label": "sulfide",
      "geometry": {"type": "rectangle_xyxy", "x1": 100, "y1": 200, "x2": 180, "y2": 260},
      "actor": "non_expert_qa",
      "note": "obvious bright sulfide missed by pseudo-label fusion"
    },
    {
      "edit_type": "binary_label_assignment",
      "target_label": "uncertain",
      "geometry": {"type": "polygon_xy", "points": [[10, 20], [40, 20], [30, 50]]},
      "actor": "non_expert_qa",
      "note": "ambiguous boundary; exclude from hard-label training"
    }
  ]
}
```

Patch application rules:

```text
target_label=sulfide          -> sulfide_mask=1, ignore_mask=0
target_label=not_sulfide      -> sulfide_mask=0, ignore_mask=0
target_label=uncertain        -> ignore_mask=1
target_label=exclude_artifact -> ignore_mask=1
```

Training loss must ignore pixels where `corrected_ignore_mask > 0`.

## Training Loop

### Phase 1. LumenStone Pretraining

1. Build a LumenStone binary label map.
2. Train binary ResUNet/U-Net or SegFormer.
3. Validate on held-out LumenStone images.
4. Save checkpoint and metrics with license/provenance notes.

Deliverables:

```text
binary_lumenstone_dataset_manifest.json
binary_sulfide_pretrain_checkpoint.pth
metrics_lumenstone_binary.json
```

### Phase 2. Official Pseudo-Label Generation

1. Run Petroscope on official images.
2. Run the pretrained binary student on official images.
3. Run the brightness/morphology baseline.
4. Fuse labels into high-confidence and ignore masks.
5. Generate QA overlays and per-image summaries.

Deliverables:

```text
official_pseudo_labels/
  sample_id/
    petroscope_sulfide_mask.png
    student_sulfide_mask.png
    heuristic_sulfide_mask.png
    pseudo_sulfide_mask.png
    ignore_mask.png
    qa_overlay.png
    pseudo_label_summary.json
```

### Phase 3. Optional Non-Expert QA

1. Launch the Streamlit QA app on precomputed pseudo-label outputs.
2. Review only high-impact disagreement regions.
3. Correct visually obvious false positives/false negatives.
4. Mark uncertain regions as `ignore`, not as hard labels.
5. Save every edit as a binary QA patch.
6. Apply patches into corrected binary masks and training manifests.

Deliverables:

```text
corrections/
  binary_sulfide_qa_patch.json
  corrected_masks/
    corrected_sulfide_mask.png
    corrected_ignore_mask.png
  training_manifest.json
```

### Phase 4. Student Fine-Tuning

1. Train from LumenStone-pretrained checkpoint.
2. Use corrected masks where available.
3. Use fused pseudo-labels with ignore masks elsewhere.
4. Keep validation split leakage-safe by file stem / source folder.
5. Export confidence heatmaps and disagreement maps.

Deliverables:

```text
binary_sulfide_final_checkpoint.pth
metrics_official_pseudo_validation.json
confidence_heatmaps/
```

### Phase 5. Downstream Ore Classifier

1. Convert `sulfide_mask` to connected components.
2. Build closing-based grain footprints.
3. Compute replacement features:
   - dark-inside ratio;
   - compactness;
   - solidity;
   - boundary complexity;
   - fragmentation/local component density;
   - area.
4. Grid-search thresholds against official image-level classes.
5. Use the official talc detector separately.

Deliverables:

```text
component_features.csv
intergrowth_thresholds.json
ore_classification_metrics.json
```

## High-Resolution Panorama Inference

Official panoramas include very large JPEGs; the current local dataset has images up to `27025 x 21227` px. Full-image inference is not practical for these inputs.

Use overlapping tiled inference:

```text
panorama
-> overlapping tiles
-> batched model inference
-> weighted stitching / vote stitching
-> full-size binary sulfide mask
-> global connected-component postprocess
```

Recommended initial settings:

```text
tile_size = 1536 or 2048
overlap = 192 or 256
batch_size = auto by available VRAM
```

Overlap is required. Without it, tile borders can cut veins, thin intergrowths, talc texture, and dark-intrusion features, producing visible seams and broken components.

Parallelization rules:

- For GPU inference, prefer one GPU process with batched tiles. Do not launch many competing inference processes on one GPU.
- Parallelize CPU-side tile decoding, preprocessing, and output writing with a worker pool.
- For multiple panoramas, shard images across machines (`gx10` / `zelda`) or queue them per GPU.
- Keep tile metadata stable: source image, tile origin, tile size, overlap, preprocessing settings, model checkpoint, device, runtime, and memory estimate.

Avoid holding full-size multi-class `float32` logits for the whole panorama. For binary sulfide detection, store compact outputs:

```text
full_mask_uint8.png
confidence_uint8.png
tile_summary.jsonl
```

If probability stitching is needed, prefer streaming or stripe-based accumulation rather than a full panorama logits buffer. `uint16` vote counts or `float16` probabilities are acceptable intermediate formats when documented.

After stitching, run a full-image postprocess:

- connected components across tile boundaries;
- morphology closing/opening on the stitched mask;
- grain-footprint reconstruction;
- replacement/dark-intrusion feature extraction.

This global pass is mandatory. Otherwise a single large sulfide inclusion split by tile boundaries can be incorrectly treated as several separate grains.

## Evaluation

Report metrics honestly by supervision source:

- LumenStone binary segmentation IoU/F1: true proxy labels.
- Official pseudo-label agreement: weak-label quality, not ground truth.
- Official image-level ordinary/fine/talcose F1: task-level calibration metric.
- Talc fraction error: only on usable official blue-line-derived talc masks.

Avoid claims like:

```text
expert-corrected sulfide masks
ground-truth official sulfide segmentation
Petroscope proves final accuracy
```

Use instead:

```text
high-confidence pseudo-labels
weakly supervised official-domain fine-tuning
image-level validation against official class folders
```

## Implementation Tasks

1. Add binary label-map conversion for LumenStone.
2. Add Petroscope-to-binary pseudo-mask exporter.
3. Add heuristic bright-sulfide mask exporter.
4. Add mask-fusion script with `ignore` support.
5. Add QA overlay generation for all intermediate masks.
6. Add binary model training script for gx10/zelda.
7. Add the Streamlit binary QA app and binary patch applier.
8. Add final tiled inference command for large official panoramas.
9. Add documentation and metric tables separating true proxy metrics from weak official-domain metrics.

## Done Definition

The plan is complete when one command can:

```text
official images
-> Petroscope / student / heuristic masks
-> fused pseudo-labels with ignore zones
-> fine-tuned binary sulfide model
-> sulfide mask and confidence heatmap for each panorama
```

and the final official pipeline consumes that `sulfide_mask` for ordinary-vs-fine component classification without claiming unavailable expert ground truth.
