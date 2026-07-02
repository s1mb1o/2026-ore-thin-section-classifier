# Domain Dataset Search for Official Shlif Task

Date: 2026-07-02

## Scope

Search for datasets relevant to the official `Скажи мне кто твой шлиф` task:

- panoramic OM images of polished sections;
- ore-class labels: ordinary, difficult/fine-intergrowth, talcose;
- possible talc-region or intergrowth supervision;
- adjacent polished-section mineral segmentation datasets for pretraining;
- SEM/XRD/automated-mineralogy datasets only where they help support the official OM task.

Initial pass recorded metadata only. Later on 2026-07-02, the official package was staged locally under `dataset/` through file-by-file public API downloads; the current local status is recorded below.

## Short Conclusion

The official Yandex package is now the primary domain dataset. Public datasets are useful mostly as pretraining/proxy material:

1. Official Yandex package: exact target distribution and labels; inspect first after download.
2. LumenStone S2/S1/S3/P1/P2: best public polished-section segmentation and panorama proxy, especially S2 for Norilsk Group sulfide mineral association.
3. Cu and FeM Zenodo datasets: good reflected-light ore/resin binary segmentation and RLM/SEM registration proxies.
4. Iron Ore Image Data: useful RLM classification/domain-augmentation source, not mask supervision.
5. USGS Wet Mountains thin-section + automated-mineralogy maps: good example of image + mineral-map pairing, but transmitted-light whole thin sections rather than reflected-light polished ore.
6. SEM datasets such as cigRockSEM and Ni-WC are secondary; use for robustness/model plumbing, not official task claims.

No better public talc/intergrowth polished-section dataset was found in this pass. The official package's `Области оталькования` folder is therefore the likely key talc-supervision asset.

Local download status:

- Path: `dataset/`
- Manifest: `dataset/_download_manifest.json`
- Verified locally: `1236` of `1236` files, `3,018,194,503` bytes, by size and SHA-256.
- Previous blocker resolved: Yandex returned `DiskResourceDownloadLimitExceededError` / `Превышен лимит скачивания` for four remaining files on 2026-07-02; retry on 2026-07-03 succeeded.
- Files recovered on retry:
  - `/Фото руд по сортам. ч1/Рядовые руды/DSCN5024.JPG`
  - `/Фото руд по сортам. ч2/рядовые/1822217 1.jpg`
  - `/Фото руд по сортам. ч2/рядовые/DSCN3086.JPG`
  - `/Фото руд по сортам. ч2/тонкие/DSCN1452.JPG`

## Official Yandex Package Metadata

Source:

- Official task data link from `docs/official/Скажи мне кто твой шлиф.md`: `https://disk.yandex.ru/d/Fo5eIM984glHaA`
- Metadata command used:

```bash
curl -L "https://cloud-api.yandex.net/v1/disk/public/resources?public_key=https%3A%2F%2Fdisk.yandex.ru%2Fd%2FFo5eIM984glHaA&limit=100"
```

Top-level folders:

| Folder | Metadata finding |
| --- | --- |
| `Панорамы` | 14 JPG files, about 1.31 GiB total. |
| `Фото руд по сортам. ч1` | Class folders: `Оталькованные руды`, `Рядовые руды`, `Труднообогатимые руды`. |
| `Фото руд по сортам. ч2` | Class folders: `оталькованные`, `рядовые`, `тонкие`. |

Detailed listing summary:

| Path | Files | Approx size | Notes |
| --- | ---: | ---: | --- |
| `/Панорамы` | 14 JPG | 1.31 GiB | Large target panoramas numbered `4.jpg` through `17.jpg`. |
| `/Фото руд по сортам. ч1/Оталькованные руды` | 42 JPG + 1 subdir | 32 MiB for listed image files | Includes nested `Области оталькования`. |
| `/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования` | 42 JPG | 79 MiB | Likely talc-area visual supervision; inspect after download. |
| `/Фото руд по сортам. ч1/Рядовые руды` | 68 JPG | 53 MiB | Class-labeled ordinary ore examples. |
| `/Фото руд по сортам. ч1/Труднообогатимые руды` | 68 JPG | 53 MiB | Class-labeled difficult ore examples. |
| `/Фото руд по сортам. ч2/оталькованные` | 87 JPG | 278 MiB | Class-labeled talcose examples. |
| `/Фото руд по сортам. ч2/рядовые` | 497 JPG | 564 MiB | Class-labeled ordinary examples. |
| `/Фото руд по сортам. ч2/тонкие` | 418 image files | 512 MiB | Mostly JPG plus one BMP; fine-intergrowth/difficult examples. |

Immediate implication:

- Treat official data as mixed classification + region/visual-supervision package, not guaranteed pixel masks.
- The four files that hit Yandex's public-resource download limit on 2026-07-02 were recovered on 2026-07-03; the local package is complete.
- Inspect whether `Области оталькования` images are overlays, crops, line annotations, or masks, and whether filenames pair one-to-one with `Оталькованные руды`.
- Build the first local manifest by class folder, filename stem, magnification tokens (`5x`, `10x`, `20x`), and panorama ID.

## Best External Datasets

### LumenStone

Source: https://imaging.cs.msu.ru/en/research/geology/lumenstone

Fit:

- Best public polished-section segmentation proxy.
- The source describes prepared polished sections from 30 CIS ore deposits and pixel-level masks.
- S2 is the closest mineral association: Layered Ultramafic Deposits / Norilsk Group with pyrrhotite, chalcopyrite, pentlandite, and magnetite.
- P1/P2 are panorama-stitching polished-section datasets and are useful for the official panoramic-image requirement.

Use:

- Pretrain or validate segmentation infrastructure before adapting to official class folders.
- Use P1/P2 for panorama tiling/stitching logic.
- Use V1 for color/condition adaptation.

Caveat:

- LumenStone classes are minerals, not official `ordinary_intergrowth` / `fine_intergrowth` / `talc`.
- License is a data usage agreement for research and citation, not a standard permissive package license; do not bundle raw data in final artifacts without permission/need.

Local status:

- Already downloaded locally under `data/external/lumenstone/`.
- Full public corpus and S2 v2 were previously verified in `docs/notes/dataset-candidates.md`.

### Cu Dataset

Source: https://zenodo.org/records/5020566

Fit:

- 121 reflected-light microscopy image/reference-mask pairs.
- Copper ore sample with sulfides, oxides, silicates, and native copper.
- Correlative RLM/SEM acquisition with registration; binary target is ore vs resin.
- CC BY 4.0 on Zenodo.

Use:

- Binary ore/background segmentation pretraining.
- Registration and RLM/SEM reference-mask handling.
- General sulfide/oxide polished ore texture exposure.

Caveat:

- Binary ore/resin labels do not solve official intergrowth/talc classes.
- Copper ore is adjacent, not Norilsk/Ni-Cu official data.

### FeM Iron Ore Dataset

Source: https://zenodo.org/records/5014700

Fit:

- 81 reflected-light microscopy image/reference-mask pairs.
- Binary ore vs embedding resin labels.
- Correlative RLM/SEM acquisition, registered to 999x756 px at 1.05 micrometer per pixel.
- CC BY 4.0 on Zenodo.

Use:

- Binary segmentation sanity checks and area-fraction plumbing.
- Reflected-light polished-section preprocessing and report tests.

Caveat:

- Iron ore concentrate; not sulfide intergrowth/talc classification.

### Iron Ore Image Data

Source: https://data.mendeley.com/datasets/6hp82tsb8g/2

Fit:

- Reflected-light microscopy images of iron ores across four grades and 10x/20x magnifications.
- TIF images, 2048x1536, CC BY 4.0.

Use:

- Ore-classification augmentation or pretraining.
- Domain-shift checks across grade and magnification.

Caveat:

- No segmentation masks; not sulfide/talc-specific.

### USGS Wet Mountains Thin Sections and Automated Mineralogy

Source: https://www.usgs.gov/data/thin-section-images-automated-mineralogy-scans-lithogeochemistry-and-nd-sr-pb-isotopic

Fit:

- USGS data release with thin-section images, automated mineralogy mineral maps, whole-rock geochemistry, and isotopic data.
- PPL/XPL entire-section images plus TESCAN TIMA automated-mineralogy scans.
- Mafic-ultramafic intrusion context is geologically closer than generic sandstone/cell microscopy.

Use:

- Reference for image + automated-mineralogy-map pairing.
- Possible training/validation design for QEMSCAN/TIMA-derived labels.

Caveat:

- Transmitted-light thin sections and automated mineralogy maps, not reflected-light polished-section ore photos.
- Not a direct official model-training source unless scope expands toward automated mineralogy map supervision.

## Secondary / Support Datasets

| Dataset | Source | Use | Caveat |
| --- | --- | --- | --- |
| cigRockSEM | https://zenodo.org/records/14988631 | SEM pore/fracture/rock-microstructure robustness and segmentation plumbing. | Mudstone/sandstone/shale SEM; not polished OM or sulfide mineral classes. |
| Ni-WC SEM segmentation | https://zenodo.org/records/17315241 | SEM segmentation and defect-like class plumbing with masks. | Additive-manufactured composite, not ore. |
| Petrographic Image Segmentation GitHub | https://github.com/jbardelli/Petrographic-Image-Segmentation | Small PPL/XPL sandstone mineral segmentation example. | GPL-3.0 code; thin-section sandstone, not ore. |
| QEMSCAN-map thin-section segmentation work | https://arxiv.org/abs/2505.17008 and https://github.com/ltracegeo/deep-mineralogical-segmentation | Method reference for training from QEMSCAN maps. | Paper/code reference more than a ready public ore dataset. |

## Recommended Next Actions

1. Generate the next local manifest with folder class, filename, extension, dimensions, file size, image mode, and EXIF if present.
2. Inspect `Области оталькования` visually and programmatically:
   - same stems as talcose images?
   - overlay vs crop vs binary/colored mask?
   - can it produce a talc-region mask or only weak supervision?
3. Build first official-data split by sample stem, not random image, to avoid leakage across near-duplicates.
4. Use external datasets only for support:
   - LumenStone: polished-section segmentation/panorama/robustness.
   - Cu/FeM: binary ore/background and RLM/SEM registration.
   - Iron Ore Image Data: grade/magnification classification augmentation.
5. Do not bundle raw external datasets into final submission unless each license/usage rule is satisfied and attribution is included.
