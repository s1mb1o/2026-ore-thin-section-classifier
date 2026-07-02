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
  - epoch 6: val sulfide IoU `0.940857`
  - epoch 7: val sulfide IoU `0.930861`
  - epoch 8: val sulfide IoU `0.944586`
  - epoch 9: val sulfide IoU `0.943122`
  - epoch 10: val sulfide IoU `0.947211`
  - epoch 11: val sulfide IoU `0.944813`
  - epoch 12: val sulfide IoU `0.946529`
  - epoch 13: val sulfide IoU `0.945835`
  - epoch 14: val sulfide IoU `0.947859`
  - epoch 15: val sulfide IoU `0.944637`
  - epoch 16: val sulfide IoU `0.948193`
  - epoch 17: val sulfide IoU `0.948684`
  - epoch 18: val sulfide IoU `0.949281`
  - epoch 19: val sulfide IoU `0.950296`
  - epoch 20: val sulfide IoU `0.947377`
  - epoch 21: val sulfide IoU `0.950462`
- Best observed so far: epoch 21, val sulfide IoU `0.950462`, val bg IoU `0.943302`, val pixel accuracy `0.972844`.
- Final benchmark summary:
  - best epoch: `26`
  - best val sulfide IoU: `0.956436`
  - val bg IoU at best: `0.950908`
  - val pixel accuracy at best: `0.976373`
  - sulfide F1 at best: `0.977733`
  - sulfide AUC at best: `0.996942`
  - HD95 mean on 512 sampled val tiles: `37.37 px`
  - final epoch 30 val sulfide IoU: `0.953216`
  - total training seconds: `7242.35`
  - average seconds per epoch: `241.41`
  - checkpoint size: `96M`
- Local mirror: `models/binary_sulfide/resunet_dataset_v0_gx10_20260703_004425/`
- Local checksum `best.pt`: `fac6799ced81f2341168607230fa0d7766fab3f12854430b9922eaa5550e7308`
- Local checksum `last.pt`: `0d815047868853202217a95fa94a6e5fb577efa5f41b0a6cbdbebf9daa117c02`

## zelda

- Current user-supplied host: `root@161.104.48.181`
- Workspace: `/root/2026_Nornikel_Hackaton_v2`
- Synced: v2 code/docs and `outputs/binary_sulfide_dataset_v0`
- Prepared venv: `/root/2026_Nornikel_Hackaton_v2/.venv`
- Rebuilt venv after recovery; installed packages include torch `2.5.1+cu121`, torchvision `0.20.1+cu121`, OpenCV `5.0.0`, numpy, transformers `5.12.1`.
- GPU recovered on retry:
  - `lspci`: Virtio GPU plus `NVIDIA Corporation Device 2684`
  - `nvidia-smi`: `NVIDIA GeForce RTX 4090`, `49140 MiB`
  - `torch.cuda.is_available()`: `True`
- CUDA sanity run passed for SegFormer-B0.
- Completed session: `tmux nornickel_v2_segformer_b0`
- Output dir: `outputs/train_segformer_b0_zelda_20260702_220225`
- Command: SegFormer-B0, 30 epochs, batch 16, AMP, `lr=6e-5`, CUDA.
- First live check after launch: process PID `1631`, GPU utilization around `94%`, GPU memory around `6117 MiB / 49140 MiB`.
- Final benchmark summary:
  - best epoch: `13`
  - best val sulfide IoU: `0.953371`
  - val bg IoU at best: `0.947638`
  - val pixel accuracy at best: `0.974712`
  - final epoch 30 val sulfide IoU: `0.951119`
  - total training seconds: `824.83`
  - average seconds per epoch: `27.49`
  - checkpoint size: `43M`
- Local mirror: `models/binary_sulfide/segformer_b0_dataset_v0_zelda_20260702_220225/`
- Local checksum `best.pt`: `6133984ab605424ef9a42a4486857ba1872fae87fa2a1fa63ebe9b49a6368162`
- Epoch metrics:
  - epoch 1: val sulfide IoU `0.902211`
  - epoch 2: val sulfide IoU `0.907983`
  - epoch 3: val sulfide IoU `0.92744`
  - epoch 4: val sulfide IoU `0.916018`
  - epoch 5: val sulfide IoU `0.935003`
  - epoch 6: val sulfide IoU `0.937818`
  - epoch 7: val sulfide IoU `0.938191`
  - epoch 8: val sulfide IoU `0.939145`
  - epoch 9: val sulfide IoU `0.948772`

Detailed benchmark note: `docs/benchmarks/01_binary_sulfide_model_benchmark.md`.

## zelda SegFormer-B1

- Completed session: `tmux nornickel_v2_segformer_b1_safe`
- Output dir: `outputs/train_segformer_b1_zelda_20260703_overnight_safetensors`
- Command: SegFormer-B1, 30 epochs, batch 16, AMP, `lr=6e-5`, CUDA.
- Training script was updated to prefer `use_safetensors=True` because zelda has torch `2.5.1+cu121` and Transformers rejects unsafe `.bin` loading for some models.
- Final benchmark summary:
  - best epoch: `16`
  - best val sulfide IoU: `0.971548`
  - val bg IoU at best: `0.967670`
  - val pixel accuracy at best: `0.984634`
  - sulfide F1 at best: `0.985569`
  - sulfide AUC at best: `0.998522`
  - HD95 mean on 512 sampled val tiles: `26.25 px`
  - final epoch 30 val sulfide IoU: `0.964032`
  - total training seconds: `1097.59`
  - average seconds per epoch: `36.59`
  - checkpoint size: `160M`
- Local mirror: `models/binary_sulfide/segformer_b1_dataset_v0_zelda_20260703_overnight_safetensors/`
- Local checksum `best.pt`: `e71ceb0d3df88b8f24473c5fb4b82678303d854a2f8b15ad1af66022dea11908`
- Local checksum `last.pt`: `03db84dbce6395cd381c2be568d9a366aeaf94cfab573ce80c34566d7a435d11`

## zelda SegFormer-B2

- Added code support for `segformer_b2` as the next quality candidate after B1.
- Completed session: `tmux nornickel_v2_segformer_b2`
- Output dir: `outputs/train_segformer_b2_zelda_20260703_overnight_safetensors`
- Command: SegFormer-B2, 30 epochs, batch 16, AMP, `lr=6e-5`, CUDA.
- Pretrained encoder: `nvidia/mit-b2`; decode head is initialized for binary sulfide segmentation.
- First launch check: GPU utilization reached `100%`, memory around `17.9 GiB / 49.1 GiB`.
- Final benchmark summary:
  - best epoch: `20`
  - best val sulfide IoU: `0.974381`
  - val bg IoU at best: `0.970874`
  - val pixel accuracy at best: `0.986181`
  - sulfide F1 at best: `0.987024`
  - sulfide AUC at best: `0.998811`
  - HD95 mean on 512 sampled val tiles: `23.57 px`
  - final epoch 30 val sulfide IoU: `0.969119`
  - total training seconds: `2352.14`
  - average seconds per epoch: `78.40`
  - checkpoint size: `320M`
- Local mirror: `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/`
- Local checksum `best.pt`: `55c31ef645cfb5c9b0b8fd91f4b9d2070e425b32ed60e23b3c15b292546b910f`
- Local checksum `last.pt`: `40cc2fa920282964d70588a9815a94915611a22f0182e97327c629220119f00c`
