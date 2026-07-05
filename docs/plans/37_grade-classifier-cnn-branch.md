# Plan 37 — Grade classifier CNN branch

- Date: 2026-07-04
- Deliverable: `scripts/train_grade_classifier.py` + `scripts/evaluate_grade_branch.py`
- Related: `docs/specs/ore-pipeline-eval-harness.md`

## Motivation

An end-to-end supervised CNN (`efficientnet_b3`) trained directly on the
grade-folder labels is the right tool for grade classification. Our pipeline is
segmentation-first and gets 0.185 (deterministic rule) / 0.747 (feature-CV). We
add a **parallel CNN grade branch** alongside our interpretable segmentation
branch (which stays for reports and per-grain intergrowth classification).

## Key data constraints (verified 2026-07-04)

1. **Talc segmentation does NOT identify the оталькованная grade.** On the 345
   eval split, `talc_fraction` for talcose images is ~0 (median 0.0000); at any
   threshold talcose one-vs-rest F1 = 0. The grade "talc > 10% by mass" is not
   visible as detectable talc regions in reflected-light OM. So talcose cannot be
   decided by our talc segmenter today.
2. **Talcose data is scarce**: 129 total / 115 deconflicted, and the fixed 345
   eval split consumes all 115 → **0 talcose left to train on** if we hold out
   the eval split.

**Decision (user, 2026-07-04):** the CNN branch classifies **ordinary ↔ fine**
only (where data is plentiful: 412 + 343 after holdout). **Talcose is deferred to
the talc segmentation branch, to be finished later.** The 3-class verdict is
gated on that future work.

## Training design (`scripts/train_grade_classifier.py`)

- **Model**: `torchvision.models.efficientnet_b3(weights=IMAGENET1K_V1)`, classifier
  head replaced with `Linear(in, num_classes)`. No new deps (timm/albumentations
  avoided; torchvision already present locally and on gx10).
- **Classes**: `ordinary_intergrowth`, `fine_intergrowth` (configurable).
- **Training pool**: manifest images with those labels, **excluding** the eval
  split paths, label-conflict paths, and any content whose sha256 matches an eval
  image; deduped by sha256. → clean held-out eval on the 345's ordinary/fine.
- **Split**: grouped train/val (group = leading specimen-id regex, else sha256
  fallback — DSCN camera names have no specimen id), ~15% val for model selection.
- **Imbalance**: class-weighted CrossEntropy (inverse-frequency).
- **Optim/sched**: AdamW (lr 3e-4, wd 1e-4), linear warmup → cosine annealing.
- **Aug** (torchvision): RandomResizedCrop(384, 0.7–1.0), H/V flip, rotation,
  ColorJitter, light blur; ImageNet normalize. Val: Resize+CenterCrop(384).
- **AMP** on CUDA; best checkpoint by **val macro-F1**.
- Outputs: `best.pt` (+ state_dict, classes, img_size, normalize), `metrics.json`,
  resolved `train_val_split.json`.

## Compute

- **gx10** (192.168.86.14, GB10, aarch64, driver 580). zelda is offline (spot).
- Env: `~/Projects/train-models/.venv/bin/python` (torch 2.12.0+cu130, tv 0.27.0).
- Repo + dataset + `outputs/*` already present on gx10.
- Flow: smoke locally on MPS → scp script to gx10 → train → pull `best.pt` into
  `models/grade_classifier/effb3_ordfine_<date>/`.

## Evaluation (`scripts/evaluate_grade_branch.py`)

- Run the trained model on the **230 ordinary/fine images of the 345 eval split**
  (held out from training) → binary macro-F1, per-class P/R/F1, confusion.
- Compare with our feature-CV (ordinary 0.72 / fine 0.72). Report as the CNN
  grade-branch result.
- 3-class hybrid number is explicitly **pending talc-seg** (do not fabricate a
  talcose F1).

## Steps

1. [done] Plan (this doc).
2. [done] Training script + local MPS smoke.
3. [done] Deploy + train on gx10; pull checkpoint.
4. [done] Eval on the held-out 230; comparison update.
5. [done] Docs: comparison note, ChangeLog, session-sync.

## Results (2026-07-04, gx10 GB10)

- Training: pool 755 (excluded 230 eval + 37 conflict + 7 eval-dup + 22 train-dup),
  train 642 (346 ord / 296 fine), internal val 113. 25 epochs, AdamW 3e-4,
  cosine+warmup, class-weighted CE, AMP. Internal-val best macro-F1 **0.955**.
- **Held-out test (230 ordinary/fine of the 345 split, excluded from training):**
  macro-F1 **0.9303**, accuracy 0.9304; ordinary F1 **0.9333** (P 0.896 / R 0.974),
  fine F1 **0.9273** (P 0.971 / R 0.887); confusion ordinary [112, 3], fine [13, 102].
- Lifts our learned ordinary/fine from feature-CV ~0.72 to ~0.93.
- Artifacts: `models/grade_classifier/effb3_ordfine_20260704/` (`best.pt`,
  `metrics.json`, `heldout_eval.{json,md}`, `train_val_split.json`).
- **Caveat:** 2-class only. A 3-class headline waits on the talcose branch (talc
  segmentation, deferred). Data gotcha fixed en route:
  gx10 initially had only the 345 eval images synced — full dataset rsynced before
  the successful run; the script now skips missing files defensively.

## Reproduce

```bash
# train (gx10)
~/Projects/train-models/.venv/bin/python scripts/train_grade_classifier.py \
  --out-dir models/grade_classifier/effb3_ordfine_20260704 \
  --img-size 384 --epochs 25 --batch-size 32 --num-workers 8 --device auto
# eval on held-out 230
python3 scripts/evaluate_grade_branch.py \
  --checkpoint models/grade_classifier/effb3_ordfine_20260704/best.pt \
  --out-json models/grade_classifier/effb3_ordfine_20260704/heldout_eval.json \
  --out-md  models/grade_classifier/effb3_ordfine_20260704/heldout_eval.md
```

## Honesty notes

- Residual leakage risk: same-аншлиф DSCN photos (no specimen id) may split across
  train/eval; unavoidable from filenames.
- The CNN branch is ordinary↔fine only; any 3-class headline must wait for the
  talc branch and must not reuse eval talcose for threshold fitting.
