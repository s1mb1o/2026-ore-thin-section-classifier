# Consolidated metrics — «Скажи мне, кто твой шлиф» (2026-07-05)

Single reference for all our numbers. GT sources and splits are labelled per row —
they are **not** all the same ruler; read the notes.

## 1. Segmentation (pixel-level, weak-label val)

### Sulfide / non-sulfide (our core model)
| model | init | val IoU | F1 | AUC | HD95 px |
| --- | --- | ---: | ---: | ---: | ---: |
| **SegFormer-B2** (deployed) | ImageNet MiT-B2 | **0.9744** | 0.9870 | 0.9988 | 23.6 |
| SegFormer-B1 | ImageNet MiT-B1 | 0.9715 | 0.9856 | 0.9985 | 26.3 |
| ResUNet (base 32) | scratch | 0.9564 | 0.9777 | 0.9969 | 37.4 |
| SegFormer-B0 | ImageNet MiT-B0 | 0.9534 | 0.9761 | 0.9962 | 33.9 |

*weak-label metrics (Otsu/agreement pseudo-labels + LumenStone proxy), model-selection criterion — not expert pixel GT.*

### Talc (non-sulfide matrix)
| model | metric | value |
| --- | --- | ---: |
| SegFormer-B0 (5-fold) | mean talc IoU / F1 | **0.644 / 0.782** |
| ResUNet (local) | val talc IoU | 0.527 |

*GT = 42 blue-contour expert images, auto-converted. Competitors: nail U-Net IoU 0.12/Dice 0.19; opium Dice 0.49.*

### Talc-fraction error (OOF, 42 annotated images) — proxy for the "±3%" criterion
| denominator | mean abs | median abs | within ±3pp |
| --- | ---: | ---: | ---: |
| image | 7.77 pp | 5.53 pp | **33%** |
| analyzed | 8.02 pp | 5.78 pp | 33% |
| non-sulfide | 11.56 pp | 8.08 pp | 19% |

*Leak-free out-of-fold (each image scored by the fold where it was in val). GT = reviewed blue-contour masks (weak, not a true expert fraction), on the talc-heavy annotated subset. `outputs/evaluations/talc_fraction_error_oof_20260705.*`*

## 2. Grade / ore-class (image-level, deconflicted 345-split, folder-label GT)

Deconflicted split = sha256 conflict+duplicate content excluded (115/class).

| approach | classes | split | F1-macro | per-class |
| --- | --- | --- | ---: | --- |
| Deterministic rule (B2 + auto-talc) | 3 | 345 | 0.185 | row 0.168 / fine 0.387 / talc 0.000 |
| **Rule (B2 + B0-talc)** ⭐ | 3 | 345 | **0.508** | row 0.207 / fine 0.464 / **talc 0.851** |
| Feature-CV (B2 + auto-talc) | 3 | 345 (5-fold) | 0.747 | row 0.719 / fine 0.721 / talc 0.800 |
| **Feature-CV (B2 + B0-talc)** | 3 | 345 (5-fold) | **0.770** | row 0.719 / fine 0.744 / talc 0.847 |
| Path B grains (bootstrap) | 3 | 345 (grouped CV) | 0.190 | row 0.086 / fine 0.483 / talc 0.000 |
| Path B grains + B0-talc | 3 | 345 (grouped CV) | 0.513 | row 0.143 / fine 0.575 / **talc 0.821** |
| **Grade-CNN ord↔fine (raw)** | 2 | 230 held-out | **0.930** | ord 0.933 / fine 0.927 |
| **Grade-CNN ord↔fine (preproc-aware)** ⭐ | 2 | 230 held-out | **0.939** | ord 0.941 / fine 0.937 |
| **Fused verdict: talc-branch ⊕ Grade-CNN** ⭐⭐ | 3 | 345 | **0.861** | row 0.867 / fine 0.865 / talc 0.851 |

**Fused verdict is the deployable full 3-class result** (`scripts/eval_grade_fusion.py`,
`outputs/evaluations/grade_fusion_20260705.*`): talcose from the B0 talc branch,
ordinary↔fine from the Grade-CNN, morphology rule as fallback. **macro-F1 0.861**
(row 0.867 / fine 0.865 / talc 0.851) — a 3–4× lift over the pure rule (0.508) on
row/fine, and **on par with competitor A (0.880)** on a leak-free 345 split (theirs
218). 211/345 decided by the CNN, 134 by the talc branch. Note: the "тип срастаний
≥90%" criterion is carried by the standalone 2-class Grade-CNN (0.93); inside the
3-class fusion the talc branch's false-positives pull row/fine to ~0.865.

### Grade-CNN 4-class benchmark (competitor A's schema)
| run | model | img | val F1-macro | ordinary | thin | talc | refractory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| effb3_4class (best) | efficientnet_b3 | 384 | **0.771** | 0.906 | 0.792 | 0.865 | 0.522 |
| nail (reference) | efficientnet_b3 | 384 | 0.791 | 0.920 | 0.870 | 0.910 | 0.470 |

*grouped-by-аншлиф val (train 926 / val 163), class-weighted CE. On par with A; better on the weak refractory (56 imgs total).*

## 3. Grade-CNN robustness (held-out 230, macro-F1)

| condition | raw | preproc-aware (p=0.5) | pp0.7+acquisition |
| --- | ---: | ---: | ---: |
| baseline | 0.930 | 0.939 | 0.939 |
| blur + noise | 0.909 | 0.944 | 0.913 |
| color shift | 0.900 | 0.917 | 0.913 |
| acquisition artifacts | 0.900 | 0.908 | **0.944** |
| UI preprocessing | 0.869 | 0.917 | 0.917 |
| **worst-case** | 0.869 | 0.908 | **0.913** |

## 4. Backends & params (pipeline benchmark)

| stage | backend | key params |
| --- | --- | --- |
| Sulfide segmentation | SegFormer-B2 (ImageNet-finetuned) | tile 1024 / stride 768, thr 0.5, resident single-load |
| Talc segmentation | SegFormer-B0 (fold_00, 5-fold) | talc-thr 0.50, clipped to non-sulfide pixels |
| Grain/grade | deterministic rule (component morphology) | min-comp-area 128 px, close-kernel 21 px, area/replacement-ratio/solidity/compactness |
| Grade-CNN branch (parallel) | efficientnet_b3 preproc-aware | img 384, ImageNet-pretrained, class-weighted CE |
| Eval split | deconflicted 345 | sha256 conflict(24 groups/48 paths)+dup excluded |
| Device | **gx10 (NVIDIA GB10, CUDA)** | resident single-load; ~full 345 run in minutes |

**Talc-backend impact (headline):** swapping the color auto-candidate for the trained **SegFormer-B0** talc model lifts the deterministic rule pipeline from macro-F1 **0.185 → 0.508**, driven entirely by talcose **F1 0.000 → 0.851** (precision 0.79 / recall 0.92). The ordinary↔fine axis stays the rule's weak spot (row 0.21 / fine 0.46) — that is what the grade-CNN (0.93) and feature-CV (0.77) address. Run: `outputs/evaluations/bench_b2_b0_rule_gx10_20260705/` (on gx10).

**Compute note:** all inference on GPU servers, not the Mac. gx10 (GB10) ran the full 345-image B2+B0 pipeline in minutes; zelda (RTX 4090 spot) was attempted for a cross-GPU check but its WAN link could not sustain the multi-GB checkpoint transfer (dataset landed, checkpoints kept dropping even chunked).

## 5. Compliance vs official acceptance criteria

| Criterion | Requirement | Our result | Verdict |
| --- | --- | --- | --- |
| Intergrowth-type classification | F1 ≥ 90% | grade-CNN ord↔fine macro-F1 **0.930/0.939** (held-out 230, folder-GT) | **✅ met** (2-class ord/fine; image-level CNN; folder-label GT, not pixel-expert) |
| Talc-fraction error | ≤ ±3% vs expert | **No reference exists** — organizers confirmed (QA #4) they cannot provide talc fractions for the blue-line images. Proxy vs weak masks: OOF median ~5.5 pp | **⛔ not gradeable** (no GT for anyone), practical proxy below target |

Honest framing for the jury: we hit F1 ≥ 90% on intergrowth type. The talc **±3% criterion is not evaluable at all** — in the 4th QA session the organizers were asked to provide the talc fraction for at least some blue-line-marked images and replied that they **do not have that information and cannot provide it**. So no expert reference exists — not for us, the jury, or the organizers. The only talc annotation is the 42 hand-drawn contours (positions, not calibrated fractions). Against those weak masks our leak-free OOF fraction error is ~5–8 pp (median 5.5), and — the part that matters for the sort — the "talc > 10%" grade decision agrees with the masks on **90%** of the 42 (talcose grade F1 0.851 in the pipeline). Bottom line: the ±3% target cannot be certified by anyone on this data; we deliver a working talc-detection branch and an honest proxy instead.

**Jury ruling (QA #4, verbatim in `docs/notes/2026-07-03-official-metrics-and-panorama-split.md`):** solutions are graded against **the team's own documented definition** of talc / оталькованная / intergrowth types ("тальк — это… то, что команда прописала"), with the **10% threshold kept**. Action: state our operational definitions explicitly in the README/deck — this converts "no GT" into a documented, defensible choice we are then graded on.

## 6. Honesty

- No expert pixel-level GT; grade GT = official grade folder per аншлиф. Segmentation metrics are weak-label.
- Splits differ across rows (2-class held-out 230 vs 3-class 345 vs 4-class grouped 163 vs nail's 218) — comparisons indicative, not controlled.
- Deployed grade branch = 2-class ord↔fine (0.93); talcose handled by segmentation/rule; 3-class full verdict pending the talc-branch fusion.
