# Talc Silicate-Support Labeling Plan

Date: 2026-07-03

## Goal

Improve talc supervision without assuming that every silicon/silicate pixel is
talc. The converter should be able to use an optional binary
silicate-support mask as evidence inside blue-line talc annotations and as hard
negative evidence outside them.

## Principle

Silicate support is a confidence signal, not a talc label. Talc-positive
training pixels must still come from the official blue-line talc candidate, with
sulfides and uncertain boundaries removed.

## Implementation

1. Keep the current default converter behavior unchanged when no silicate mask
   is supplied.
2. Add an optional `--silicate-mask-dir` argument with masks matched by image
   stem, mirroring `--sulfide-mask-dir`.
3. For each sample, write additional QA/training masks:
   - `silicate_support_mask.png`: supplied binary support mask, resized to the
     image if needed.
   - `silicate_supported_talc_mask.png`: blue-line talc candidate minus
     sulfides, intersected with silicate support.
   - `silicate_unsupported_talc_mask.png`: blue-line talc candidate minus
     sulfides, outside silicate support; treat as uncertain, not positive.
   - `talc_positive_core_mask.png`: eroded conservative talc-positive core for
     training.
   - `silicate_hard_negative_mask.png`: silicate-supported pixels outside the
     talc candidate and away from candidate borders.
4. When silicate support is supplied, use supported talc as `final_talc_mask`
   and add unsupported talc candidate pixels into `ignore_mask`.
5. Store counts, support source, and support fraction in
   `conversion_summary.json` and `manifest.json`.
6. Add focused unit tests for supported talc, unsupported uncertain talc, and
   silicate hard negatives.

## Expected Effect

The training set can use:

```text
talc_positive_core_mask.png       -> positive talc pixels
silicate_hard_negative_mask.png   -> not_talc hard negatives
ignore_mask.png                   -> uncertain/markup/sulfide-overlap pixels
```

This should reduce the main failure mode where the model learns
`dark/silicate matrix = talc`, while still preserving manual QA over the full
blue-line candidate.

## Risks

- A weak silicate detector can suppress true talc. This is why unsupported
  candidate pixels are marked `ignore`, not `not_talc`.
- Silicate support outside annotations is only a hard negative relative to the
  current annotation set; it should be used with sampling caps so it does not
  overwhelm positives.
