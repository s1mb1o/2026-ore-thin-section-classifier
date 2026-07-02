# Official Metrics and Panorama Split Clarification

Date: 2026-07-03

Source: user-provided organizer clarification in the project thread.

## Clarification

Organizer guidance:

- Automated scoring will not be used for the hackathon solution check.
- The target metrics to pay attention to while training models are:
  - segmentation: IoU and Hausdorff distance;
  - classification: F1 and AUC.
- These are production-solution target metrics, not a request to overfit the provided dataset.
- Teams should not try to reach `100%` on the provided data.
- Panoramas may be used as a test set, but the recommended evaluation set is a balanced sample from several classes.
- In the provided dataset, panoramas are unannotated and unclassified images.

## Implications for This Project

Current `binary_sulfide_dataset_v0` metrics are useful for model selection, but incomplete:

- current benchmark has IoU and pixel accuracy;
- next benchmark should add Hausdorff distance or HD95 for segmentation masks;
- downstream ore/image classification should report F1 and AUC;
- validation and test splits should be balanced by class where class labels exist;
- panoramas should be used for demo, performance, and optional unlabelled stress testing unless a separate annotation/classification pass is created.

Do not present the current SegFormer-B0 IoU result as final production accuracy. Present it as a weak-label baseline for choosing a sulfide segmentation checkpoint.

## Evaluation Plan Update

For the next measurable iteration:

1. Keep `binary_sulfide_dataset_v0` as weak-label baseline only.
2. Extend evaluation code to emit:
   - IoU;
   - Hausdorff distance or HD95;
   - F1 for binary segmentation thresholded at the chosen operating point;
   - AUC from probability maps where available.
3. Build a balanced image-level validation table from official class folders for:
   - ordinary / row ore;
   - fine / difficult ore;
   - talcose ore.
4. Use panoramas separately:
   - tiled inference performance;
   - visual QA overlays;
   - confidence heatmaps;
   - stress testing on large unannotated images.
5. If panoramas are later annotated or assigned class labels, record that as a new dataset version and do not mix it silently with the current unlabelled panorama set.
