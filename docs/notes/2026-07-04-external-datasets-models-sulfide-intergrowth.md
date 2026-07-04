# External datasets & models for sulfide-inclusion segmentation + intergrowth typification

Date: 2026-07-04
Scope: web scan for reusable datasets/pretrained models for the ore task
"Скажи мне, кто твой шлиф" — specifically:
- segment sulfide inclusions in optical (reflected-light) polished sections;
- classify intergrowth type:
  - **обычные срастания** — coarse, isolated sulfides, minimal replacement by a
    gray/dark phase (e.g. magnetite) → *ordinary* (рядовая) ore marker;
  - **тонкие срастания** — sulfides heavily replaced by a non-ore phase →
    *hard-to-process* (труднообогатимая) ore marker.

## TL;DR
- **Segmentation foundation exists and matches our exact mineralogy.** Use it.
- **Intergrowth-type / processability labels do NOT exist off the shelf.** That layer
  is a downstream feature-engineering + classification step we build on top of masks.
  The most relevant paper (Korshunov/MSU) explicitly lists intergrowth typification and
  processability assessment as *future work / under development*.

## Primary hit — LumenStone dataset + petroscope (MSU imaging lab)
This is the best match found; it is Russian, open, and covers Norilsk Cu-Ni sulfide ore.

- **Dataset page**: https://imaging.cs.msu.ru/en/research/geology/lumenstone
- **Paper (open, EN)**: Korshunov et al., "From visual diagnostics to deep learning:
  automatic mineral identification in polished section images", *Mining Science and
  Technology (Russia)* — https://mst.misis.ru/jour/article/view/974
- **Library (code + pretrained model)**: https://github.com/xubiker/petroscope (GPL-3.0,
  Python ≥3.10, PyTorch)
- **Contact**: Alexander Khvostikov, khvostikov@cs.msu.ru

### Why it fits
- **LumenStone S2 = "Norilsk Group" sulfide copper-nickel ores**: 39 images
  (≈23–37 train / 6–12 test), **3396×2547 px**, **5 mineral classes**
  (pyrrhotite, pentlandite, chalcopyrite + magnetite + generalized non-metallic/
  non-ore class), **pixel-level masks**. This is exactly our target mineral system.
- The per-class palette includes **magnetite** and a **non-metallic (non-ore) class** —
  i.e. exactly the "gray/dark replacing phase" and "non-ore phase" our intergrowth
  definitions depend on. So a mask from this model directly yields the geometry needed
  to measure replacement/isolation.
- Other subsets: S1 (84 imgs, Berezovskoe complex ores: galena/sphalerite/chalcopyrite/
  bornite/fahlore), S3 (35 imgs, anisotropic: arsenopyrite/covellite, incl. XPL rotations),
  V1 (color-adaptation), P1 (765–1552 overlapping imgs for panorama stitching).
- Download: **Yandex Disk** links on the dataset page (S2 v1 ≈242 MB, v2 ≈419 MB).
- **License**: research use with citation. No dataset-hosted pretrained weights.

### Reported model performance (LumenStone S1+S2)
- Best model: **PSPNet + ResNet18** encoder, class-balanced sampling.
  Overall **IoU 0.88**, pixel accuracy **0.96** over 9 minerals + non-metallic class.
- Per-class IoU: pyrite 0.964, pyrrhotite 0.928, bornite 0.938, sphalerite 0.922,
  non-metallic 0.912, galena 0.905, chalcopyrite 0.899, **pentlandite 0.790**.
  (Pentlandite is the weakest — worth noting for our Cu-Ni case.)
- petroscope also ships a **ResUnet** pretrained on **LumenStone S1v1** (7 classes) via
  pip/GitHub releases; PSPNet is the stronger reported architecture.
- petroscope extras we can reuse: illumination/distortion **calibration** and **panorama
  stitching**. It does **NOT** do intergrowth/texture analysis — segmentation only.

## Secondary / adjacent resources
- **SulfideNet** — deep model for binary sulfide segmentation, but on **RGB drill-core
  photos**, not polished sections. Dice ≈0.82, 91.9% acceptance in a prospective study.
  Different modality; public availability **unconfirmed**. Useful as prior art for the
  binary sulfide-vs-rest step, not as a drop-in for reflected-light micrographs.
- **DeepLabv3+ on reflected-light Cu/Fe ore** (Iglesias et al., *Minerals Engineering*
  2021, ScienceDirect S0892687521002363): >90% acc; introduces correlative microscopy to
  generate reference masks. General segmentation methodology.
- **Copper-ore mineralogy in flotation pulp via DL + optical microscopy**
  (ScienceDirect S0892687523004958, 2023): predicts particle size + mineralogy incl.
  chalcopyrite — closest thing to *process-mineralogy* output, but flotation-pulp domain.
- **Automatic ore texture analysis for process mineralogy** + intergrowth-characterization
  literature: these give the *quantitative intergrowth-typing methodology* (locking degree,
  contact/perimeter ratios, association) we'd implement — methods, not datasets/models.
- **Nature Scientific Data 2025** "A Photomicrographic Dataset of Rocks…"
  (s41597-025-05879-9): rock/mineral classification dataset; less aligned (classification,
  not our reflected-light sulfide segmentation).

## Recommended plan of attack
1. **Reuse LumenStone S2 + petroscope** to get per-pixel masks of
   {chalcopyrite, pentlandite, pyrrhotite, magnetite, non-ore} on our shlifs. Start from
   the pretrained model; fine-tune on our labels if domain gap shows (esp. pentlandite).
2. **Build the intergrowth-type classifier ourselves** — no public labels exist. From each
   segmented sulfide grain/cluster derive features:
   - grain size / equivalent diameter, count of isolated blobs;
   - fraction of sulfide perimeter in contact with magnetite vs non-ore vs other sulfide;
   - internal replacement fraction (magnetite/non-ore area embedded within/veining the
     sulfide);
   - shape complexity (perimeter²/area, solidity).
   Threshold or train a small classifier → {обычные срастания / тонкие срастания} →
   {рядовая / труднообогатимая} ore marker.
3. Cross-check our labeling scheme with the Korshunov group (they flag this as their own
   next step; possible collaboration / label alignment).

## Open questions / caveats
- Need to confirm LumenStone S2's exact class list and whether "non-metallic" is a single
  bucket or splits gangue vs magnetite (magnetite appears listed separately — verify on
  download).
- petroscope is **GPL-3.0** — fine for internal/hackathon use; check compatibility if we
  redistribute weights or link it into shipped code.
- SulfideNet public availability not verified — do not assume downloadable.
