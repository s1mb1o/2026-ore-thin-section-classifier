# Targeted OM Datasets and Models for Official Ore Classes

Date: 2026-07-02

## Scope

Targeted internet search for optical-microscopy datasets and models matching three official task operations:

- выделение сульфидных фаз, светлых областей, на фоне силикатной/оксидной матрицы, темных/серых областей;
- классификация срастаний по степени замещения нерудной фазой;
- детекция талька как рассеянной темной фазы в нерудной матрице.

This note complements the broader searches in:

- `docs/notes/2026-07-02-domain-datasets-search.md`
- `docs/notes/2026-07-02-training-datasets-and-models-search.md`

## Short Verdict

No public OM dataset/model found in this pass is a drop-in replacement for the official Yandex package.

Most useful external material by task:

| Official operation | Best public support | Fit | Main caveat |
| --- | --- | --- | --- |
| Light sulfides vs dark/gray matrix | LumenStone S2/S1, Petroscope ResUNet, Cu/FeM RLM binary datasets, DeepLabv3+ ore/resin work | Good for segmentation pretraining, tiling, domain adaptation, and binary bright-ore separation plumbing | Labels do not match `ordinary_intergrowth` / `fine_intergrowth` / `talc`; Cu/FeM mostly ore/resin, not ore/matrix |
| Intergrowth by non-ore replacement | Ore microscopy liberation/locking literature, automated OM/OIA work, official class folders | Best implemented as a component-level interpretable classifier calibrated on official image-level labels | I found no open pixel-level OM dataset with `ordinary vs fine sulfide intergrowth` labels |
| Talc as dark dispersed phase | Official `Области оталькования` folder is the practical source | Likely the only task-specific region supervision found | I found no public polished-section OM talc segmentation dataset or open pretrained talc detector |

Practical implication: external data should support robustness and feature learning, while final claims should be trained/calibrated on official data.

## High-Value Sources

### LumenStone and Petroscope

Sources:

- https://imaging.cs.msu.ru/en/research/geology/lumenstone
- https://github.com/xubiker/petroscope
- https://isprs-archives.copernicus.org/articles/XLIV-2-W1-2021/113/2021/isprs-archives-XLIV-2-W1-2021-113-2021.html
- https://journals.rcsi.science/2500-0632/article/view/350750

Relevant facts:

- LumenStone contains polished-section ore microscopy subsets with pixel-level masks.
- S2 is the closest geological proxy: Norilsk Group layered ultramafic deposits with pyrrhotite, chalcopyrite, pentlandite, and magnetite.
- P1/P2 are panorama-stitching polished-section datasets, useful for the official panorama workflow.
- Petroscope provides code and a ResUNet segmentation model path trained on LumenStone S1v1; the GitHub README reports mean IoU around 0.8373 on S1v1.
- The 2025 LumenStone/MISIS article reports a universal polished-section segmentation direction with IoU up to 0.88 / PA up to 0.96 for nine minerals, plus color adaptation, panorama construction, and fast annotation methods.

Use:

- Pretraining or proxy validation for bright sulfide/mineral segmentation.
- Color adaptation and active-learning ideas for small official labels.
- Panorama tiling/stitching and patch-balanced sampling.

Do not claim:

- Direct solution of `ordinary_intergrowth`, `fine_intergrowth`, or `talc`.

License/provenance note:

- Petroscope GitHub page shows GPL-3.0. LumenStone is a research dataset with its own usage/citation expectations, not a permissive bundled-data source.

### Cu Dataset

Source:

- https://zenodo.org/records/5020566

Relevant facts:

- 121 pairs of reflected-light microscopy RGB images and binary reference masks.
- Copper ore with sulfides, oxides, silicates, and native copper.
- SEM-derived reference masks after correlative RLM/SEM registration.
- Zenodo page states CC BY 4.0.
- Linked paper reports DeepLabv3+ variants with mean overall accuracy 90.56% and F1 92.12% on this dataset.

Use:

- Binary segmentation proxy for bright ore/mineral particles versus non-target background/resin.
- Domain-shift and cross-validation smoke for reflected-light polished ore images.

Caveat:

- Binary target is ore vs embedding resin; it is not a mask for silicate/oxide matrix replacement inside sulfide grains.

### FeM Dataset

Source:

- https://zenodo.org/records/5014700

Relevant facts:

- 81 reflected-light microscopy RGB image/reference-mask pairs.
- Iron ore concentrate, mostly hematite/quartz with minor magnetite/goethite.
- SEM-derived binary reference masks after RLM/SEM registration.
- Zenodo page states CC BY 4.0.
- Linked paper reports DeepLabv3+ variants with mean overall accuracy 91.43% and F1 93.13% on this dataset.

Use:

- Binary segmentation and area-fraction plumbing.
- Robustness test for reflected-light dark/gray non-ore components.

Caveat:

- Iron ore concentrate; no sulfide intergrowth or talc class semantics.

### DeepLabv3+ Ore/Resin Segmentation and Domain Adaptation

Sources:

- https://www.sciencedirect.com/science/article/abs/pii/S0892687521002363
- https://www.preprints.org/manuscript/202412.0572

Relevant facts:

- The 2021 Minerals Engineering paper targets semantic segmentation of opaque and non-opaque minerals from epoxy resin in reflected-light microscopy.
- It used four datasets from copper and iron ores and reports accuracy/F1 above 90%, up to 94% in some datasets.
- The 2024 preprint studies domain adaptation for the same ore/resin RLM task and reports large F1 gains in some source-target combinations.

Use:

- Strong model-pattern evidence for a binary stage.
- Domain adaptation warning: models trained on one ore/sample/imaging setup can fail on another; official data calibration is mandatory.

Caveat:

- Task is ore/resin or opaque/non-opaque/resin, not official intergrowth/talc classification.

### Automated OM and OIA Literature

Sources:

- https://www.sciencedirect.com/science/article/abs/pii/S0892687522005064
- https://www.mdpi.com/2076-3263/6/2/30
- https://www.zeiss.com/microscopy/en/applications/raw-materials-and-industrial-rd/mining.html
- https://msaweb.org/wp-content/uploads/2022/07/Craig_Vaughan_Chptr_11.pdf

Relevant facts:

- Reflected optical microscopy is a lower-cost route for automated quantitative mineralogy, especially for opaque minerals such as sulfides and oxides.
- The 2022 review highlights transparent gangue detection/quantification as a major OM limitation.
- Ore microscopy/OIA can compute mineral quantification and geometry metrics such as grain size and liberation.
- Craig and Vaughan's ore microscopy chapter gives a useful liberation/interlocking texture taxonomy; replacement/interpenetration textures are tied to difficult liberation.
- ZEISS describes automated mineral classification and liberation/association measurement as commercial microscopy workflows, but this is not an open dataset.

Use:

- Justification for component-level features: sulfide component area, dark-inclusion ratio, boundary complexity, skeleton/porosity, and adjacency to matrix.
- Language for explaining intergrowth-class logic to judges.

Caveat:

- Literature and commercial workflows do not provide the exact official labels or open pretrained model.

### Res-UNet Ensemble and Recent Polished-Section Segmentation Papers

Sources:

- https://www.mdpi.com/2075-163X/14/12/1281
- https://link.springer.com/article/10.1007/s42461-025-01205-4

Relevant facts:

- The 2024 Res-UNet ensemble paper reports mIoU 91.65 across nine categories and discusses difficulty with transparent/similar minerals, minority classes, and liberation-bias implications.
- The 2025 Mask R-CNN polished-section paper is method-relevant for mineral photometry images.

Use:

- Model family references for optional dense 4-class upgrade.
- Support for ensemble or loss-function handling of rare classes such as talc.

Caveat:

- I did not find open weights or an openly reusable dataset from these pages during this pass.

### RoImAI, Petro-SAM, SegmentEveryGrain, and MicroNet

Sources:

- https://www.nature.com/articles/s44172-025-00565-5
- https://github.com/Jsf826/RoImAI
- https://arxiv.org/abs/2604.14805
- https://github.com/zsylvester/segmenteverygrain
- https://github.com/nasa/pretrained-microscopy-models

Relevant facts:

- RoImAI is a foundation-model direction for rock thin sections, with 30,336 microscopy images and about two million particles, but mostly transmitted/polarized thin-section workflows and no ready release/weights on the checked GitHub page.
- Petro-SAM is a 2026 SAM-based petrographic thin-section method for grain-edge and lithology segmentation, but the checked arXiv page did not expose a direct code/data path.
- SegmentEveryGrain is an Apache-2.0 SAM2 + U-Net workflow for grain instance segmentation, with interactive cleanup and fine-tuning guidance.
- NASA MicroNet provides generic pretrained microscopy encoders under an MIT repository, useful when target labels are scarce.

Use:

- Annotation assist and boundary refinement, especially for talc masks and sulfide components.
- Generic feature extraction or self-supervised/pretrained encoder fallback.

Caveat:

- None are direct polished Ni-Cu sulfide/talc OM classifiers.

## Recommended Implementation Interpretation

1. Binary sulfide stage:
   - Start with official images, but bootstrap with LumenStone S2/S1 and Cu/FeM.
   - Use brightness/reflection heuristics as a baseline, then train or fine-tune SegFormer/ResUNet/DeepLab-style binary segmentation.
   - Validate with official held-out images and measure IoU/HD95 where masks or derived masks exist.

2. Intergrowth stage:
   - Do not wait for a nonexistent public intergrowth dataset.
   - Segment sulfide connected components first.
   - For each component compute replacement features: non-ore/dark-hole ratio, component solidity, perimeter/area, skeleton width distribution, boundary roughness, internal matrix adjacency.
   - Calibrate thresholds or a small classifier on official image-level labels (`рядовые` vs `тонкие` / `труднообогатимые`), then report image-level F1.

3. Talc stage:
   - Use official `Области оталькования` as the primary talc supervision.
   - Convert blue-line annotations into filled masks, remove line pixels from training inputs, and create QA overlays.
   - Train a small talc-vs-non-talc detector only inside the non-sulfide matrix; use hard negatives from dark non-ore matrix and artifacts.
   - Report talc-fraction absolute error on held-out annotated examples.

4. Optional upgrade:
   - Pseudo-label official class-folder photos after the interpretable pipeline is stable.
   - Fine-tune a dense 4-class model (`background`, `ordinary_intergrowth`, `fine_intergrowth`, `talc`) only if it beats the interpretable baseline on official validation.

## Bottom Line

For the final hackathon path, use public OM datasets as support, not as the target:

- LumenStone/Petroscope: best polished-section proxy and model baseline.
- Cu/FeM/DeepLabv3+: best open reflected-light binary segmentation evidence.
- Automated OM/liberation literature: best justification for component-level intergrowth features.
- Official `Области оталькования`: primary talc data; no public talc OM replacement found.
