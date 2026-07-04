# Talc Model Benchmark

Date: 2026-07-04

## Scope

This benchmark evaluates the trained talc segmentation model on the reviewed
talc-mask workspace:

- reviewed masks: `outputs/talc_blue_line_conversion/samples/*/reviewed/reviewed_talc_mask.png`
- fold run: `outputs/talc_segformer_folds/segformer_b0_full_20260703`
- benchmark output: `outputs/benchmarks/talc_segformer_b0_full_image_20260704`
- script: `scripts/benchmark_talc_model.py`
- model family: SegFormer-B0, 5 image-level folds
- samples: 42 held-out reviewed talc images
- local runtime: MPS, full-image tiled inference, tile/stride `1024 / 768`, batch `1`

The model is evaluated only on analyzed non-sulfide pixels for segmentation
metrics. Talc-fraction error uses analyzed non-ignored pixels as denominator,
matching the ore-fraction reporting intent.

Important caveat: the target masks are non-expert reviewed masks derived from
the blue-line talc workflow, not independent expert geological ground truth.

## Results

| Source | Talc IoU | Talc F1 | Precision | Recall | Pixel acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| SegFormer-B0 held-out folds | `0.6410` | `0.7812` | `0.7729` | `0.7897` | `0.8472` |
| Original blue-line converter baseline | `0.1577` | `0.2725` | `0.9751` | `0.1584` | `0.7079` |

The full-image result is consistent with the earlier tile-calibrated 5-fold
summary, which reported mean talc IoU `0.644191` and mean F1 `0.782301`.

## Talc Fraction Error

- SegFormer-B0 MAE: `8.551` percentage points.
- Median absolute error: `5.813` pp.
- P90 absolute error: `14.907` pp.
- Signed bias: `+0.649` pp.
- Images within +/-3 pp: `0.214` of samples.
- Blue-line baseline MAE: `24.389` pp.

This is a large improvement over the draft blue-line converter, but it is not
yet good enough for a strict +/-3 percentage-point talc-fraction claim.

## Boundary Metrics

- Hausdorff mean: `427.63` px.
- Hausdorff median: `334.14` px.
- HD95 mean: `177.91` px.
- HD95 median: `105.27` px.

These numbers should be treated as review-routing indicators rather than final
geological boundary accuracy because the reviewed masks are noisy and some
failures are large-region fraction errors.

## Fold Summary

| Fold | Samples | Threshold | Talc IoU | Talc F1 | Fraction MAE pp | Within +/-3 pp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | `0.50` | `0.5658` | `0.7227` | `10.753` | `0.200` |
| 1 | 9 | `0.50` | `0.6388` | `0.7796` | `7.034` | `0.333` |
| 2 | 8 | `0.40` | `0.6271` | `0.7708` | `10.401` | `0.125` |
| 3 | 8 | `0.35` | `0.7304` | `0.8442` | `6.604` | `0.250` |
| 4 | 7 | `0.55` | `0.6485` | `0.7868` | `7.467` | `0.143` |

## Worst Fraction Errors

| Sample | GT fraction | Pred fraction | Error pp | IoU | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `DSCN5180` | `0.3019` | `0.7744` | `47.251` | `0.3893` | `0.5605` |
| `DSCN4720` | `0.0874` | `0.3348` | `24.741` | `0.2259` | `0.3686` |
| `DSCN3042` | `0.4906` | `0.2772` | `21.335` | `0.5528` | `0.7120` |
| `DSCN4755` | `0.2969` | `0.4857` | `18.883` | `0.5475` | `0.7076` |
| `2550376-1 5x` | `0.3334` | `0.1835` | `14.990` | `0.4280` | `0.5994` |
| `DSCN3056` | `0.3179` | `0.1764` | `14.153` | `0.4081` | `0.5797` |
| `DSCN4714` | `0.5413` | `0.6802` | `13.886` | `0.7383` | `0.8495` |
| `2550374-2 10х` | `0.0869` | `0.2190` | `13.207` | `0.1466` | `0.2557` |

## Reproduction

```bash
./.venv/bin/python scripts/benchmark_talc_model.py \
  --folds-dir outputs/talc_segformer_folds/segformer_b0_full_20260703 \
  --out-dir outputs/benchmarks/talc_segformer_b0_full_image_20260704 \
  --device auto \
  --batch-size 1 \
  --overwrite
```

The run writes:

- `outputs/benchmarks/talc_segformer_b0_full_image_20260704/summary.json`
- `outputs/benchmarks/talc_segformer_b0_full_image_20260704/summary.md`
- `outputs/benchmarks/talc_segformer_b0_full_image_20260704/per_sample_metrics.csv`

## Recommendation

Use the trained talc model for segmentation overlays and as a stronger source
than the blue-line draft masks, but do not claim the talc fraction is within
+/-3 pp yet. The next step is threshold/fraction calibration against held-out
images, then a small review queue for the high-error samples above.
