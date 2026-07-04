# Plan 43: Ore Pipeline Talc Cluster Areas

## Goal

Copy the talc annotation tool's cluster-area visualization into the v2 ore pipeline with persistent settings, per-run configuration, final-mode overlays, and report metrics.

## Steps

1. Document feature behavior in `docs/ui/v2/specs/ore-pipeline-talc-cluster-areas-v0.1.md`.
2. Add backend defaults and payload normalization:
   - `DEFAULT_TALC_CLUSTERIZATION`
   - `normalize_talc_clusterization_payload`
   - settings persistence under `settings.talc_clusterization`
   - run persistence under `run.talc_clusterization`
3. Add cluster generation:
   - compute local-density cluster mask from talc mask and analyzed mask
   - save `masks/talc_cluster_mask.png`
   - add `display.talc_cluster_overlay`
   - enrich summary and metrics with cluster area and effective parameters
4. Wire API payloads:
   - `/api/runs/start`
   - `/api/runs/{run_id}/prepare`
   - `/api/runs/{run_id}/start`
   - batch shared settings
5. Add UI controls:
   - Settings defaults card
   - `Configuration...` button below Metadata
   - configuration modal with radius/min local talc/opacity
   - include configuration in start, prepare, and batch settings payloads
6. Add viewer behavior:
   - final legend checkbox for talc cluster areas in left and right legends
   - percentage in legend row
   - draw cluster overlay on final layers only
7. Update tests:
   - backend normalization and mask calculation
   - run payload/display/metrics assertions
   - static UI assertions
8. Update handoff docs, run focused tests, restart the local service, and commit as a topic.

## Acceptance Criteria

- New runs contain `talc_clusterization` with effective values.
- Completed runs contain `masks.talc_cluster`, `display.talc_cluster_overlay`, and cluster area summary fields.
- `final` view can toggle talc cluster visibility independently from talc pixels.
- Settings defaults survive reload.
- Configuration dialog defaults come from Settings and are included in Start/Apply flows.
- Metrics CSV and PDF report path include talc cluster area through shared metric rows.

