# SegFormer-B0 Binary Sulfide Checkpoint

Date mirrored locally: 2026-07-03

Source host:

- `root@161.104.48.181:/root/2026_Nornikel_Hackaton_v2/outputs/train_segformer_b0_zelda_20260702_220225`

Dataset:

- `outputs/binary_sulfide_dataset_v0/manifest.json`

Files:

- `best.pt`: best validation sulfide IoU checkpoint, epoch 13
- `last.pt`: final epoch 30 checkpoint
- `metrics.json`: best validation sulfide IoU summary
- `train_log.csv`: full 30-epoch training log

Metrics:

- best val sulfide IoU: `0.9533708053109677`
- final epoch 30 val sulfide IoU: `0.951119`

Checksums:

- `best.pt`: `6133984ab605424ef9a42a4486857ba1872fae87fa2a1fa63ebe9b49a6368162`
- `last.pt`: `fa64b00fe460ad67c4b150622d51ef33cbbe5aeadbbd82fb3952282498263cce`

Note: `models/*` is intentionally git-ignored. This README is a local artifact note, not a tracked repo document.
