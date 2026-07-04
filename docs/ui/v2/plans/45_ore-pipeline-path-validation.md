# Plan 45: Ore Pipeline Path Validation Hardening

Date: 2026-07-04

## Problem

The v2 ore pipeline store has a validated helper for existing run directories, but some run and upload helpers read `run.json` or `upload.json` through direct `self.runs_dir / run_id` and `self.uploads_dir / upload_id` path joins. A crafted ID containing `..` can escape the intended root if a sibling directory contains the expected JSON file.

## Scope

- Harden run and upload ID handling in `apps/ore_pipeline_web.py`.
- Preserve existing valid run/upload behavior and immutable run semantics.
- Add focused regression tests in `tests/test_ore_pipeline_web.py`.
- Do not broaden this change into authentication, large upload streaming, or file-download streaming.

## Implementation Steps

1. Add a shared root-relative directory validator for persisted store IDs.
2. Route `_read_run()` through `_existing_run_dir()` and introduce `_existing_upload_dir()` for `_read_upload()`.
3. Replace direct run/upload write-directory joins in request-facing methods with the validated helpers where the ID is user-provided.
4. Add regression tests for traversal rejection on `run_payload`, `cancel_run`, `upload_payload`, and upload-derived write operations.
5. Run focused web tests and restart the local UI service so the running app uses the fix.

## Acceptance Criteria

- `../` run IDs are rejected before any read or write outside `outputs/ore_pipeline_ui/runs`.
- `../` upload IDs are rejected before any read or write outside `outputs/ore_pipeline_ui/uploads`.
- Normal upload, preprocessing, run, history, and file-list paths continue to work.
- Targeted tests pass.
