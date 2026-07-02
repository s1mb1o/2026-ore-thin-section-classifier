# Training Datasets and Models Search for Official Shlif Task

Date: 2026-07-02

## Scope

Second-pass internet search for external training datasets and pretrained/domain models relevant to the official `Скажи мне кто твой шлиф` task.

The search focuses on:

- reflected-light or polished-section ore microscopy;
- thin-section or petrographic classification/segmentation datasets that can help pretraining;
- domain-specific segmentation/classification models;
- SEM/materials-microscopy datasets only where they help pipeline robustness, not official-score claims.

The user is already downloading the official Yandex package, so this pass does not download external archives or model weights.

## Short Conclusion

No public source found in this pass is a better training target than the official Yandex package for the required ore classes:

- ordinary intergrowth;
- fine/difficult intergrowth;
- talcose ore / talc region.

Best practical external sources:

1. Petroscope + LumenStone remains the strongest actionable polished-section proxy. It has public code, public LumenStone datasets, and a domain ResUNet model path, but the mineral labels do not match the official classes.
2. RoImAI is the most relevant recent foundation-model direction for rock thin sections, but the public value is currently mainly architectural/paper reference unless usable weights/data access are confirmed.
3. DeepCarbonate, MUMDMC2025, and Thin_Section are useful image-level thin-section pretraining sources, not direct mask supervision for ore/talc.
4. Petro-SAM and SegmentEveryGrain are useful method references for SAM/SAM2-assisted annotation and grain-boundary work, not direct final classifiers.
5. OD_MetalDAM, Ni-WC SEM, and synthetic SEM steel datasets/models are useful only as SEM/materials-microscopy robustness or plumbing checks.

## High-Value Findings

### Petroscope / LumenStone Models

Sources:

- https://github.com/xubiker/petroscope
- https://github.com/xubiker/petroscope/blob/main/petroscope/segmentation/models/resunet/model.py
- https://imaging.cs.msu.ru/en/research/geology/lumenstone

Finding:

- Petroscope is a Python package for microscopic geological images.
- Its segmentation module is explicitly designed around LumenStone, including class labels and class-balanced patch sampling.
- The README reports a ResUNet trained on LumenStone S1v1 for 7 segmentation classes, with mean IoU around `0.8373` and void-border mean IoU around `0.8506`.
- The current ResUNet registry in `model.py` includes a downloadable `s1s2_resnet34_x05` checkpoint URL for `S1v2_S2v2_x05.pth`.

Fit:

- Best available external polished-section model family.
- Good baseline/pretraining source for ore mineral segmentation, tiling, balancing, and inference infrastructure.

Caveat:

- LumenStone classes are minerals, not official classes.
- The repository page and package metadata expose inconsistent license signals (`GPL-3.0` on GitHub page, `MIT` classifier in `pyproject.toml`), so redistribution and final-submission packaging need a manual license check.

Recommended action:

- Keep Petroscope/LumenStone as a proxy baseline and possible feature extractor.
- Do not claim it directly solves `ordinary` / `fine intergrowth` / `talc`.
- If used in the final artifact, include exact model/data provenance and license notes.

### RoImAI

Sources:

- https://www.nature.com/articles/s44172-025-00565-5
- https://github.com/Jsf826/RoImAI

Finding:

- The article describes RoImAI as a rock thin-section foundation model for segmentation, particle identification, and lithology report generation.
- Reported dataset scale: `30,336` microscopy images and about two million rock particles from `17` regions.
- The architecture combines a Swin Transformer-based segmentation model, boundary refinement, multimodal SPI/OPI image features, hierarchical particle identification, and report generation.
- The public GitHub repository exists and contains `classification`, `segmentation`, and `generating_report.py`, but the checked page showed no release, package, or obvious published weights.
- The article license is Creative Commons Attribution-NonCommercial-NoDerivatives 4.0; data/model availability needs separate confirmation before reuse.

Fit:

- Strong method reference for explaining why a domain foundation model is plausible for petrographic images.
- Potential source to monitor for weights or code reuse if the repository becomes complete enough.

Caveat:

- Mostly transmitted-light rock thin sections, not reflected-light polished sulfide ore.
- Not currently a drop-in training source unless data/weights are confirmed accessible and license-compatible.

Recommended action:

- Cite as a related-work direction in the deck or backup, not as a dependency in the runnable submission.
- Recheck repository releases only if there is spare time after the official-data pipeline is working.

### DeepCarbonate

Source:

- https://www.nature.com/articles/s41597-026-06633-5
- Dataset DOI in article: https://doi.org/10.5281/zenodo.18061204
- Code availability link in article: https://github.com/ai4geology/DeepCarbonate

Finding:

- DeepCarbonate is a public carbonate thin-section image dataset and benchmark.
- It contains `55,786` images, `22` carbonate classes, three light modes, and about `33.2 GB` total size.
- The article describes ImageNet-style train/validation/test organization and benchmarks ResNet, VGG, DenseNet, MobileNet, and EfficientNet.
- The article says code snippets are available through the `ai4geology/DeepCarbonate` GitHub repository.

Fit:

- Useful for image-level classifier pretraining and microscopy domain augmentation.
- Good source for training robust thin-section encoders when official labels are scarce.

Caveat:

- Carbonate petrography, not Ni-Cu sulfide ore.
- Mostly classification benchmark; not a mask dataset for official three-color segmentation.
- Article is CC BY-NC-ND; the Zenodo record/license should be checked before download or redistribution.

Recommended action:

- Lower priority than LumenStone and official data.
- Use only for optional self-supervised/image-level pretraining if the official classifier overfits.

### MUMDMC2025

Source:

- https://www.nature.com/articles/s41597-025-05879-9

Finding:

- Menoufia University Machine Learning Dataset for Minerals Classification 2025 provides `14,400` high-resolution photomicrographs.
- Five mineral classes are balanced at `2,880` images each: biotite, hornblende, plagioclase, potassium-feldspar, and quartz.
- Images cover PPL/XPL conditions and 360-degree rotation sequences at 5-degree increments.
- The article states data releases follow CC BY 4.0 and GitHub processing scripts are under MIT licensing.

Fit:

- Useful for mineral-classification pretraining and augmentation under controlled optical variation.
- Good for teaching the model invariance to rotation/polarization-related appearance changes.

Caveat:

- Granite rock-forming minerals, not ore minerals or polished reflected-light sections.
- No official ore-class mask semantics.

Recommended action:

- Consider only as auxiliary classification pretraining or contrastive/self-supervised material.

### Thin_Section Captioning Dataset / App

Sources:

- https://github.com/stalyn314/Thin_Section
- https://paulhcleverley.com/2025/06/03/automatic-description-of-rock-thin-sections-a-web-application-open-source/

Finding:

- The project/paper describes an automatic rock thin-section description app.
- The blog summary reports `5,600` rock thin-section images across `14` rock categories, with PPL and XPL images and textual descriptions.
- The GitHub repository contains a `ThinSection_Dataset` directory and model/application code.

Fit:

- Potential source for image-level pretraining or report-generation demos.
- Useful if we want language/report examples for thin sections.

Caveat:

- Rock-type captioning/classification, not ore-class segmentation.
- Repository page did not show a clear license in the checked view; verify before using.

Recommended action:

- Do not prioritize before official data or LumenStone.
- Keep as an optional report/captioning reference.

### Petro-SAM

Source:

- https://arxiv.org/abs/2604.14805

Finding:

- Petro-SAM is a 2026 arXiv method for petrographic thin-section segmentation.
- It adapts SAM for grain-edge segmentation and lithology semantic segmentation, with a Merge Block for seven polarized views.

Fit:

- Useful idea source for SAM-assisted annotation or boundary refinement.
- Relevant to any expert-correction / interactive mask workflow.

Caveat:

- The checked arXiv page did not expose a code or dataset link.
- Thin-section multi-polarization setup does not match official reflected-light polished ore images.

Recommended action:

- Keep as related work only.
- Do not block implementation waiting for this model.

### SegmentEveryGrain

Source:

- https://github.com/zsylvester/segmenteverygrain

Finding:

- Apache-2.0 Python package for detecting grains or grain-like objects.
- Uses SAM 2.1 for high-quality object outlines and a U-Net-style first-pass segmentation to create prompts.
- Includes editing/cleanup functions and a path to add corrected masks back into a dataset.

Fit:

- Useful for annotation acceleration and mask-cleanup workflow design.
- The SAM2 + first-pass-prompt pattern is relevant to talc-region or intergrowth-region annotation.

Caveat:

- Sedimentary/grain segmentation, not sulfide ore class segmentation.
- It detects instances rather than solving official class semantics.

Recommended action:

- Consider the workflow pattern for a judge-visible correction tool or offline annotation helper.
- Avoid presenting it as an ore classifier.

### OD_MetalDAM

Sources:

- https://github.com/ari-dasci/OD-MetalDAM
- https://huggingface.co/datasets/Voxel51/OD_MetalDAM

Finding:

- Metallography dataset from additive manufacturing steels.
- Public GitHub page lists `42` labeled grayscale SEM images and `164` unlabeled images.
- Hugging Face card describes pixel-level masks for five classes: matrix, austenite, martensite/austenite, precipitate, and defects.
- Hugging Face card lists MIT license.

Fit:

- Useful for materials-microscopy segmentation plumbing and defect-like mask handling.
- Good SEM-side smoke/proxy dataset.

Caveat:

- SEM additive manufacturing steel, not reflected-light ore.
- Too small and different-domain for final official model claims.

Recommended action:

- Keep as SEM support only; do not mix into official OM ore training unless explicitly doing domain-robust pretraining experiments.

### Ni-WC SEM Dataset and Pretrained HF Model

Sources:

- https://zenodo.org/records/17315241
- https://huggingface.co/imranlabs/sem-microstructure-segmentation

Finding:

- Zenodo dataset has SEM crops/masks for additively manufactured Ni-WC metal-matrix composites.
- Hugging Face model card exposes a U-Net with ResNet34 encoder, one-channel input, five output classes, and a reported validation mIoU of `0.876` for its dataset.

Fit:

- Useful for pretrained SEM segmentation smoke tests and model-loading examples.
- Close enough to materials microscopy for pipeline robustness, not geology.

Caveat:

- SEM composite microstructure, not OM polished ore.
- The model's five classes are not official classes.

Recommended action:

- Use only as optional SEM/model-plumbing reference.

### Synthetic SEM Steel Precipitates

Source:

- https://huggingface.co/datasets/research-centre-rez/synthetic-sem-steel-precipitates

Finding:

- Hugging Face dataset of synthetic SEM-like steel microstructures with binary precipitate masks.
- Dataset card reports CC BY 4.0, 10K-100K size range, and procedural generator source code.
- Access requires accepting repository conditions on Hugging Face.

Fit:

- Useful for controlled segmentation and robustness tests.
- Could support synthetic-to-real discussion if needed.

Caveat:

- Synthetic steel SEM, not ore OM.
- Requires Hugging Face gated-condition acceptance.

Recommended action:

- Low priority for official classifier; keep for robustness experiments only.

### Apollo Rock Thin Section Classifier

Source:

- https://github.com/esa/apollo_rock_thin_section_classifier

Finding:

- MIT-licensed repository for Apollo thin-section breccia/basalt classification.
- Uses NASA PDS, Lunar Institute Data, and Virtual Microscope sources.
- Mentions downloadable trained models through Weights & Biases.

Fit:

- Useful example of sample-grouped splits, thin-section metadata handling, and lightweight classifier training.

Caveat:

- Lunar breccia/basalt, not ore.
- Not a relevant training source for official classes.

Recommended action:

- Treat as classifier-engineering reference only.

## Practical Ranking for This Repository

| Rank | Source | Use now? | Why |
| ---: | --- | --- | --- |
| 1 | Official Yandex package | Yes, after download | Only exact target distribution and class semantics. |
| 2 | LumenStone + Petroscope | Yes, already local | Strongest polished-section proxy and existing model baseline. |
| 3 | DeepCarbonate | Maybe | Large thin-section classification pretraining, but carbonate/not ore. |
| 4 | MUMDMC2025 | Maybe | Controlled mineral-classification pretraining, but granite minerals/not ore. |
| 5 | RoImAI | Reference only | Strong model direction; public weights/data not confirmed. |
| 6 | SegmentEveryGrain / Petro-SAM | Reference/helper only | Annotation and SAM-assisted segmentation patterns. |
| 7 | OD_MetalDAM / Ni-WC / synthetic SEM | Support only | Materials-microscopy robustness, not official OM training. |
| 8 | Thin_Section / Apollo classifier | Low | Report/classifier references, not official task data. |

## Recommended Next Actions

1. Do not download more external data until the official Yandex package is unpacked and inspected.
2. Build the official-data manifest first: folder class, filename, dimensions, magnification tokens, and whether `Области оталькования` pairs with talcose images as masks/overlays/crops.
3. Use LumenStone/Petroscope as the first external baseline:
   - pretrain/validate tiling and segmentation infrastructure;
   - keep official class mapping separate from LumenStone mineral labels.
4. If official labels are too small, test image-level encoder pretraining on DeepCarbonate/MUMDMC only after the official baseline exists.
5. Use RoImAI, Petro-SAM, and SegmentEveryGrain in the presentation as evidence that petrographic foundation/SAM-assisted workflows are credible, not as runnable dependencies.
6. Keep SEM datasets out of the official OM training path unless the final pitch explicitly includes SEM support as a secondary demo.
7. Before final packaging, create a license/provenance table for every external dataset/model actually used.
