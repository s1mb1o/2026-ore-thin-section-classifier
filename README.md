# Nornickel Hackathon v2: Official Ore Classifier

Clean workspace for the official `Скажи мне, кто твой шлиф` task.

The goal is a narrow optical-microscopy pipeline:

```text
panorama image
-> binary sulfide segmentation
-> component features
-> ordinary_intergrowth / fine_intergrowth classification
-> talc detection
-> official ore-class rule and report artifacts
```

This v2 directory intentionally avoids the old broad QC assistant surface. The old repository remains the source for archived plans, prior experiments, and reusable snippets, but new P0 implementation should live here.

## Layout

```text
AGENTS.md / CLAUDE.md
ChangeLog.md
ResearchLog.md
SMOKE_TESTS.md
docs/
  official/   # official task page copy
  plans/      # selected implementation plans
  specs/      # official requirement mapping
  notes/      # selected source/research notes
apps/         # Streamlit QA tools
scripts/      # dataset and training utilities
src/ore_classifier/
outputs/      # generated artifacts, ignored by git
models/       # local pointers/config only; HF cache stays outside repo
dataset -> ../2026_Nornikel_Hackaton/dataset
```

## Current Data Source

`dataset` is a relative symlink to the verified dataset in the original project:

```text
../2026_Nornikel_Hackaton/dataset
```

The source manifest in the old repository verified `1236/1236` files and about `3.0 GB` of official data. Keep the symlink unless there is a concrete reason to copy the dataset.

## Core Docs

- `docs/plans/25_standalone-ore-classifier-project.md`
- `docs/plans/26_weak-supervision-sulfide-binary-model.md`
- `docs/notes/talc-blue-line-conversion.md`
- `docs/specs/official-tz-solution-map.ru.md`
- `docs/official/Скажи мне кто твой шлиф.md`
- `SMOKE_TESTS.md`

## Implemented Blocks

### Talc Blue-Line Conversion And QA

The talc annotation path is implemented in the v2 layout:

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

Review UI:

```bash
streamlit run apps/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

The current full run contains `42` samples with status counts:
`31 candidate_ok`, `9 needs_manual_review`, and
`2 sulfide_overlap_review_required`.

## Next Implementation Steps

1. Build a dataset manifest from the symlinked official data.
2. Review and accept/fix the generated talc masks from `outputs/talc_blue_line_conversion`.
3. Implement binary sulfide pseudo-label generation: Petroscope teacher, LumenStone/pretrained student, and brightness morphology baseline.
4. Implement the Streamlit binary QA app in `apps/`.
5. Train and validate the binary sulfide model on `gx10` / `zelda`.
6. Add component-level ordinary/fine classification and talc fraction rule.
