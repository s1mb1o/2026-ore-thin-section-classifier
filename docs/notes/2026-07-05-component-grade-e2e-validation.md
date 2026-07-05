# Component-grade opt-in variant: E2E validation

Date: 2026-07-05

Sample: `dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG`

Environment:

- Python: `/opt/homebrew/opt/python@3.14/bin/python3.14`
- Runtime device selected by inference: `mps`
- Image size: `2272 x 1704`

## Commands

Default judged path: ML sulfide + ML talc + rule component grading.

```bash
python3 scripts/run_ore_pipeline.py \
  --image 'dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG' \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --talc-checkpoint outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt \
  --talc-threshold 0.50 \
  --out-dir outputs/e2e_validation/20260705_component_grade_default \
  --device auto \
  --batch-size 2 \
  --preview-max-side 1200
```

Opt-in component-grade variant:

```bash
python3 scripts/run_ore_pipeline.py \
  --image 'dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG' \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --talc-checkpoint outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt \
  --talc-threshold 0.50 \
  --component-model models/component_grade/hgb_weak100_nomag_20260705/model.joblib \
  --magnetite-prep \
  --out-dir outputs/e2e_validation/20260705_component_grade_variant \
  --device auto \
  --batch-size 2 \
  --preview-max-side 1200
```

## Results

| path | talc source | component model | magnetite prep | components | verdict | ordinary/fine sulfide fraction |
|---|---|---|---|---:|---|---|
| default | `ml_model` | `null` | `null` | 154 | `hard_to_process_ore` | `0.197149 / 0.793644` |
| variant | `ml_model` | `models/component_grade/hgb_weak100_nomag_20260705/model.joblib` | decision recorded, `applied=false`, reason `massive_ore(bright=45%)` | 154 | `hard_to_process_ore` | `0.000000 / 0.990793` |

Artifact verification checked `pipeline_summary.json`, binary sulfide summary,
sulfide/analyzed/talc masks, talc model summary, ore summary, component features
CSV, and intergrowth preview for both runs. All required artifacts existed and
were non-empty; the verifier printed `assertions=ok`.

The default path stayed on rule component grading (`component_model=null`,
`magnetite_prep=null`). The opt-in path loaded the component model, relabeled all
154 components through the model, recorded magnetite-prep provenance, and produced
all expected masks, summaries, CSV, and previews.

Local warning: loading `model.joblib` emits scikit-learn `InconsistentVersionWarning`
because the artifact was pickled with sklearn `1.9.0` while this local interpreter
has sklearn `1.8.0`. The model loaded and the run completed, but the deployment
environment should keep sklearn pinned to the artifact version when possible.

Magnetite-prep caveat: this sample exercised the decision/provenance path but did
not trigger the second darkened sulfide pass (`applied=false`). No clean local
`applied=true` trigger sample was identified during this validation.
