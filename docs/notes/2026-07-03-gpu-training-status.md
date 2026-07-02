# GPU Training Status

Date: 2026-07-03

## Binary Sulfide Dataset v0

- Manifest: `outputs/binary_sulfide_dataset_v0/manifest.json`
- Tile size / stride: `512 / 384`
- Total tiles: `8536`
- Split: `6948` train, `1588` val
- Sources:
  - LumenStone: `124` sources, `2976` tiles
  - Official image-level heuristic masks: `240` sources, `5560` tiles
- Sulfide class ids use the Petroscope/LumenStone mineral ids for sulfides and exclude magnetite/hematite/native gold.

## gx10

- Host: `ashmelev@192.168.86.14`
- Workspace: `/home/ashmelev/Projects/2026_Nornikel_Hackaton_v2`
- Venv used: `/home/ashmelev/Projects/lenta-synth/.venv/bin/python`
- Active session: `tmux nornickel_v2_resunet`
- Output dir: `outputs/train_resunet_gx10_20260703_004425`
- Command: ResUNet, 30 epochs, batch 16, AMP, `base_channels=32`, CUDA.
- Latest observed metrics:
  - epoch 1: val sulfide IoU `0.912475`
  - epoch 2: val sulfide IoU `0.924633`
  - epoch 3: val sulfide IoU `0.931152`
  - epoch 4: val sulfide IoU `0.932628`
  - epoch 5: val sulfide IoU `0.934942`

## zelda

- Current user-supplied host: `root@161.104.48.181`
- Workspace: `/root/2026_Nornikel_Hackaton_v2`
- Synced: v2 code/docs and `outputs/binary_sulfide_dataset_v0`
- Prepared venv: `/root/2026_Nornikel_Hackaton_v2/.venv`
- Installed packages include torch `2.5.1+cu121`, torchvision `0.20.1+cu121`, OpenCV, numpy, transformers.
- GPU recovered on retry:
  - `lspci`: Virtio GPU plus `NVIDIA Corporation Device 2684`
  - `nvidia-smi`: `NVIDIA GeForce RTX 4090`, `49140 MiB`
  - `torch.cuda.is_available()`: `True`
- CUDA sanity run passed for SegFormer-B0.
- Active session: `tmux nornickel_v2_segformer_b0`
- Output dir: `outputs/train_segformer_b0_zelda_20260702_220225`
- Command: SegFormer-B0, 30 epochs, batch 16, AMP, `lr=6e-5`, CUDA.
- First live check after launch: process PID `1631`, GPU utilization around `94%`, GPU memory around `6117 MiB / 49140 MiB`.
- Latest observed metrics:
  - epoch 1: val sulfide IoU `0.902211`
  - epoch 2: val sulfide IoU `0.907983`
  - epoch 3: val sulfide IoU `0.92744`
  - epoch 4: val sulfide IoU `0.916018`
  - epoch 5: val sulfide IoU `0.935003`
  - epoch 6: val sulfide IoU `0.937818`
  - epoch 7: val sulfide IoU `0.938191`
  - epoch 8: val sulfide IoU `0.939145`
  - epoch 9: val sulfide IoU `0.948772`
