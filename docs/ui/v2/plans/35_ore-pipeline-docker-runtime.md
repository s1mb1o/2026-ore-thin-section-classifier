# Ore Pipeline Docker Runtime Plan

Date: 2026-07-03

## Scope

Create a minimal Docker runtime for the v2 ore pipeline GUI so it can be launched on the Nornickel VM as a browser-accessible service.

## Implementation Steps

1. Add a Docker-specific dependency set with only the packages required by `apps/ore_pipeline_web.py` in heuristic mode.
2. Add a Dockerfile under `docker/ore-pipeline-ui/` that copies UI, shared source, heuristic segmentation, and scripts, but excludes dataset, models, and generated outputs via `.dockerignore`.
3. Add an entrypoint that maps environment variables to the existing `ore_pipeline_web.py` CLI and creates the persistent workspace directory.
4. Add a root Compose file that builds the image, publishes `8080`, persists `outputs/ore_pipeline_ui`, and mounts `models` read-only for optional checkpoint use.
5. Document build/run commands and smoke expectations in `COMMANDS.md`, `SMOKE_TESTS.md`, and the v2 UI docs index.
6. Add a focused unit test that verifies the Docker artifacts keep the intended runtime contract.

## Acceptance Criteria

- `docker compose -f docker-compose.ore-pipeline-ui.yml up --build` starts the UI on `http://<vm-host>:8080/workspace`.
- The Docker context excludes `dataset`, `outputs`, and model checkpoints.
- The container defaults to heuristic backend and does not require GPU packages.
- App state survives container restart through the mounted workspace directory.
- A focused test covers the Dockerfile, entrypoint, compose file, and ignore rules.
