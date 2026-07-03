# ResUNet Binary Sulfide Checkpoint

Date: 2026-07-03

This local mirror contains the ResUNet architecture-diversity baseline for binary `sulfide / not_sulfide` segmentation.

## Source Run

- Host: gx10 `ashmelev@192.168.86.14`
- Remote workspace: `/home/ashmelev/Projects/2026_Nornikel_Hackaton_v2`
- Remote output dir: `outputs/train_resunet_gx10_20260703_004425`
- Dataset manifest: `outputs/binary_sulfide_dataset_v0/manifest.json`
- Model: `resunet`
- Epochs: `30`
- Best epoch: `26`

## Metrics

- Best val sulfide IoU: `0.956436`
- Val background IoU at best: `0.950908`
- Val pixel accuracy at best: `0.976373`
- Sulfide F1: `0.977733`
- Sulfide AUC: `0.996942`
- Hausdorff mean on 512 sampled val tiles: `92.30 px`
- HD95 mean on 512 sampled val tiles: `37.37 px`

Extended metrics are saved at `outputs/evaluations/resunet_best_eval_metrics.json`.

## Checksums

- `best.pt`: `fac6799ced81f2341168607230fa0d7766fab3f12854430b9922eaa5550e7308`
- `last.pt`: `0d815047868853202217a95fa94a6e5fb577efa5f41b0a6cbdbebf9daa117c02`

## Caveat

ResUNet is slower and weaker than SegFormer-B1/B2 on this weak-label validation split. Keep it as a sanity baseline, not as the demo default.
