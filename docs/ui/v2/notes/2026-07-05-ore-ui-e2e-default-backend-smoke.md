# Ore UI E2E Default Backend Smoke - 2026-07-05

Local service: `http://127.0.0.1:63589`, restarted with `/opt/homebrew/opt/python@3.14/bin/python3.14`.

## Why the service was restarted

The first default-backend E2E attempt failed for all three samples because the service was running under Xcode Python 3.9. The ML subprocess failed in `scripts/infer_binary_sulfide.py`:

```text
TypeError: zip() takes no keyword arguments
```

The app pipeline uses `zip(..., strict=True)`, so the local service must be launched with Python 3.10+; after restart with Python 3.14.4, the same E2E path completed.

## Live Defaults

Observed from `/api/settings`, `/api/status`, and `/api/runtime/test`:

- sulfide segmentation: `ml`, SegFormer-B2 checkpoint loaded on `mps`;
- talc segmentation: `ml`, SegFormer-B0 checkpoint loaded on `mps`, threshold `0.5`;
- grain classification: `heuristic` by default;
- Grade-CNN checkpoint exists, but is not used by default.

Default runtime probe passed in `17.961 s`. `grain_classification` reported `ore_grain_heuristics`.

Non-mutating `grain_backend=ml` runtime probe also passed: EfficientNet-B3 Grade-CNN loaded on `mps` in `2.965 s`.

## Default E2E Runs

All runs used saved defaults with preprocessing disabled and no runtime override.

| Expected folder class | Image | Run | Result | Match | Notes |
| --- | --- | --- | --- | --- | --- |
| `row_ore` | `dataset/Фото руд по сортам. ч2/рядовые/1822099 val 3.jpg` | `run_20260705_004539_487080000_5a361226` | `hard_to_process_ore` | no | heuristic grain stage estimated `87.8%` fine intergrowth; Grade-CNN absent |
| `hard_to_process_ore` | `dataset/Фото руд по сортам. ч2/тонкие/69 1.jpg` | `run_20260705_004554_789324000_1d80a057` | `hard_to_process_ore` | yes | talc `0.0%`, fine intergrowth `92.3%` |
| `talcose_ore` | `dataset/Фото руд по сортам. ч2/оталькованные/1822101 1.jpg` | `run_20260705_004606_993920000_2ace504a` | `talcose_ore` | yes | talc `30.7%` |

Observed model provenance in all default runs:

```text
runtime.backend = ml
runtime.talc_backend = ml
runtime.grain_backend = heuristic
models.binary_sulfide.backend = ml
models.talc.backend = ml_model
models.grain_classification.backend = ore_grain_heuristics
grade_branch_present = false
```

## Grade-CNN Override E2E Runs

The same images were rerun with a per-run runtime override `grain_backend=ml`; Settings were not mutated.

| Expected folder class | Run | Rule class | Fused class | Verdict source | Grade-CNN output | Match |
| --- | --- | --- | --- | --- | --- | --- |
| `row_ore` | `run_20260705_004710_374146000_b5b6a385` | `hard_to_process_ore` | `row_ore` | `grade_cnn` | `row_ore`, `99.27%` | yes |
| `hard_to_process_ore` | `run_20260705_004725_707074000_d6fb6689` | `hard_to_process_ore` | `row_ore` | `grade_cnn` | `row_ore`, `58.93%` | no |
| `talcose_ore` | `run_20260705_004737_927911000_aaaa50f0` | `talcose_ore` | `talcose_ore` | `talc_branch` | `hard_to_process_ore`, `100.0%` | yes |

With the override, `models.grain_classification.backend = ml` and `grade_branch_present = true`. The fused 3-class verdict uses talc branch first for talcose ore; otherwise it uses Grade-CNN for ordinary vs fine.

## Smoke Commands

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 -m py_compile apps/ore_pipeline_web.py scripts/infer_binary_sulfide.py scripts/run_ore_pipeline.py
/opt/homebrew/opt/python@3.14/bin/python3.14 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
```

Result: `60` tests passed, `1` optional OpenAPI validator test skipped.

Live route/API smoke after E2E:

```text
/ -> /workspace 200
/workspace 200
/settings 200
/status 200
/history 200
/api 200
/api/status 200
/api/settings 200
/api/openapi.json 200
```

Final live status:

```text
health = ok
backend = ml
talc_backend = ml
grain_backend = heuristic
active_jobs = none
```
