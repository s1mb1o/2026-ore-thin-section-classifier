# Ore Pipeline Status Page v0.1

Date: 2026-07-03

## Scope

This spec applies to the v2 ore pipeline UI:

```text
apps/ore_pipeline_web.py
```

It adds a read-only `Status` page for operational diagnostics on the local GUI
host or Nornickel VM.

## Goals

- Add a direct-loadable `/status` slug page.
- Add a read-only `/api/status` endpoint.
- Show current CPU load.
- Show GPU load when an NVIDIA GPU is visible through `nvidia-smi`; otherwise
  show a clear not-detected state.
- Show RAM usage.
- Show Flash/disk usage for the configured UI workspace.
- Show history size, run count, series count, upload size, and active jobs.
- Show the configured model/source for binary sulfide segmentation and talc
  detection, including talc checkpoint and threshold when talc ML is active.
- Show health checks and an overall `ok`, `warning`, or `error` state.
- Show recent system events and HTTP access events.
- Keep the page localized in Russian and English.

## Non-Goals

- Do not add new Python runtime dependencies such as `psutil`.
- Do not manage or stop jobs from the Status page.
- Do not expose host secrets, environment variables, or process command lines.
- Do not expose arbitrary host OS logs such as `/var/log/system.log`.
- Do not poll heavy directory-size scans continuously in the background.

## Data Contract

`GET /api/status` returns:

```json
{
  "schema_version": "ore-pipeline-status-v0.1",
  "generated_at": "2026-07-03T00:00:00+00:00",
  "app": {
    "started_at": "2026-07-03T00:00:00+00:00",
    "uptime_seconds": 42.0,
    "backend": "heuristic",
    "checkpoint": null,
    "checkpoint_exists": false,
    "talc_backend": "heuristic",
    "talc_checkpoint": null,
    "talc_checkpoint_exists": false,
    "talc_threshold": 0.5,
    "models": {
      "binary_sulfide": {
        "backend": "heuristic",
        "checkpoint": null,
        "role": "sulfide/non-sulfide segmentation"
      },
      "talc": {
        "backend": "heuristic_candidate",
        "configured_backend": "heuristic",
        "checkpoint": null,
        "threshold": null,
        "role": "talc detection"
      }
    },
    "workspace_dir": "outputs/ore_pipeline_ui"
  },
  "health": {
    "overall": "ok",
    "checks": [
      {"key": "workspace_writable", "status": "ok", "message": "..."}
    ]
  },
  "cpu": {
    "logical_cpus": 8,
    "load_average_1m": 1.0,
    "load_average_5m": 0.8,
    "load_average_15m": 0.7,
    "load_percent_1m": 12.5
  },
  "gpu": {
    "available": false,
    "source": "nvidia-smi",
    "message": "nvidia-smi not found",
    "devices": []
  },
  "ram": {
    "total_bytes": 34359738368,
    "available_bytes": 17179869184,
    "used_bytes": 17179869184,
    "used_percent": 50.0,
    "source": "sysconf"
  },
  "flash": {
    "path": "outputs/ore_pipeline_ui",
    "total_bytes": 1000000,
    "used_bytes": 500000,
    "free_bytes": 500000,
    "used_percent": 50.0,
    "free_percent": 50.0
  },
  "history": {
    "runs_total": 10,
    "batches_total": 2,
    "run_status_counts": {"complete": 9, "prepared": 1},
    "batch_status_counts": {"complete": 2},
    "runs_size_bytes": 123,
    "batches_size_bytes": 45,
    "uploads_size_bytes": 67,
    "history_size_bytes": 168,
    "total_workspace_size_bytes": 235,
    "active_runs": [],
    "active_batches": []
  },
  "logs": {
    "limit": 80,
    "system": [
      {
        "timestamp": "2026-07-03T00:00:00+00:00",
        "level": "info",
        "message": "service initialized",
        "details": {"backend": "heuristic"}
      }
    ],
    "access": [
      {
        "timestamp": "2026-07-03T00:00:00+00:00",
        "client": "127.0.0.1",
        "method": "GET",
        "path": "/workspace",
        "status": 200,
        "size_bytes": "-"
      }
    ]
  }
}
```

## Health Semantics

- `error` if the workspace is not writable.
- `error` if ML backend is selected and the configured checkpoint is missing.
- `error` if talc ML is selected and the configured talc checkpoint is missing.
- `error` if Flash free space is below 3%.
- `warning` if Flash free space is below 10%.
- `error` if RAM available is below 5%.
- `warning` if RAM available is below 12%.
- `warning` if CPU one-minute load exceeds 200% of logical CPU capacity.
- `warning` while runs or series are active.
- Overall health is the worst status among checks.

## UI Contract

- The header shows a `Status` / `Статус` tab.
- `/status` opens without needing a current upload or run.
- The page hides the workflow left sidebar, like History, Series, and Settings.
- The top card row shows health, CPU, GPU, RAM, Flash, history size, runs,
  series, backend, active model/source summary, and uptime.
- The health table lists all checks.
- The storage table lists run history, series history, uploads, and total
  workspace size.
- The system log panel lists bounded in-process app events such as service
  startup, run/series lifecycle events, edit recalculations, cancellations,
  failures, and request errors.
- The access log panel lists bounded in-process HTTP events with timestamp,
  method, sanitized path, status, response size when available, and client.
- Both log panels show newest entries first and an empty state if no entries
  are available.
- The `Refresh` button re-fetches `/api/status`.
