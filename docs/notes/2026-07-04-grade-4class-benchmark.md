# Grade 4-class benchmark (ordinary / thin / talc / refractory)

- Date: 2026-07-04
- Script: `scripts/train_grade_classifier.py --four-class`
- Artifacts: `models/grade_classifier/effb3_4class_20260704/` (`best.pt`, `metrics.json`, `train_val_split.json`)
- Purpose: replicate competitor A's (nail) 4-class grade schema on our data/harness.

## Setup

- 4-class labels from folders: `ordinary` (рядовые), `thin` (ч2/тонкие),
  `refractory` (труднообогатимые), `talc` (оталькованные grade only — the 42
  "Области оталькования" blue-contour annotations are excluded).
- 4-class-aware sha256 dedup + drop label-conflict content (70 conflict + 21 dup paths dropped).
- Pool 1089 → grouped-by-аншлиф train/val (leak-free): train 926 / val 163.
- efficientnet_b3 @384, ImageNet-pretrained, class-weighted CE (8–9× imbalance),
  AdamW cosine+warmup, AMP. Plain (no preprocessing/acquisition train-time aug),
  for a fair comparison to nail. Metric = internal grouped-val (as nail).
- Trained on gx10 (GB10), 25 epochs.

## Results

| run | model | img | val F1-macro | ordinary | thin | talc | refractory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| effb3_4class (best @ep7) | efficientnet_b3 | 384 | **0.7710** | 0.9059 | 0.7917 | 0.8649 | 0.5217 |
| effb3_4class (final @ep25) | efficientnet_b3 | 384 | 0.7502 | 0.9357 | 0.7742 | 0.8108 | 0.4800 |
| nail 4-class (reference) | efficientnet_b3 | 384 | 0.7910 | 0.9200 | 0.8700 | 0.9100 | 0.4700 |

## Read

- On par with competitor A (0.771 vs 0.791), same weak class — **refractory**
  (ours 0.52 vs their 0.47). refractory has only **56** images total (~8 in val),
  so its F1 is noisy for both teams; it is the true bottleneck of the 4-class task.
- We edge them on refractory; slightly behind on thin/talc; ~equal on ordinary.
- Not the same ruler: our val = 163 (grouped-by-аншлиф, dedup + conflict-dropped);
  nail's val = 218. Indicative, not a controlled head-to-head.
- Note: our production grade branch collapses thin+refractory into one "fine"
  class (2-class ordinary↔fine reaches 0.93); this 4-class run is a benchmark to
  match nail's schema, not the deployed head.

## Reproduce

```bash
python3 scripts/train_grade_classifier.py --four-class \
  --out-dir models/grade_classifier/effb3_4class_20260704 \
  --img-size 384 --epochs 25 --batch-size 32 --num-workers 8 --device auto
```
