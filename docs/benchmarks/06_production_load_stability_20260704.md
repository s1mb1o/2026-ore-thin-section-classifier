# Production Load & Stability Assessment — nornickel-ai-hackathon.alola.ru

**Date:** 2026-07-04
**Target:** `https://nornickel-ai-hackathon.alola.ru/workspace` (serving `apps/ore_pipeline_web.py`)
**Method:** External load testing with `ab` (Apache Bench) over the public HTTPS endpoint, from a single client host. Bounded ramps; stopped at the overload point. No crash observed — degradation is by latency, and the server recovered fully after each burst.

## Architecture discovered

- **Edge:** Caddy (Go) terminating TLS, serving HTTP/2 (advertises HTTP/3 `h3`), enforcing **HTTP Basic auth** (`www-authenticate: Basic realm="restricted"`) before any request reaches the Python app.
- **Backend:** `apps/ore_pipeline_web.py` — Python stdlib `ThreadingHTTPServer` + `BaseHTTPRequestHandler`. Thread-per-connection, **no thread-pool cap**, GIL-bound.
- **App-level password:** DISABLED on this deployment (`/api/auth/status` → `password_enabled: false`). Auth is entirely Caddy's basic auth.

## Results

### Edge (Caddy), unauthenticated 401s
| Scenario | RPS | p50 | Notes |
|---|---|---|---|
| keep-alive, c=50 | ~3950 | 8ms | Healthy; connection reuse cheap |
| new TLS conn each, c=50 | ~275 | 172ms | TLS handshake is the edge bottleneck; Go absorbs it without failure |

### Backend (Python app), authenticated, keep-alive
| Endpoint | Size | RPS ceiling | Behavior under load |
|---|---|---|---|
| `/api/batches` | 74 B | ~325 | General GIL/threading ceiling |
| `/api/settings` | 990 B | ~235 | — |
| **`/api/status`** | **26 KB** | **~82** | **Bottleneck. This is the endpoint the UI polls continuously.** |
| `/workspace` | 423 KB | ~70 | ~30 MB/s bandwidth; served `Cache-Control: no-store` (uncacheable) on every hit |

### `/api/status` concurrency ramp (the weak point)
| Concurrency | RPS | p50 | max |
|---|---|---|---|
| c=10 | 76 | 133ms | 337ms |
| c=50 | 83 | 575ms | 3.5s |
| c=100 | 80 | 984ms | 12s |
| c=200 | — | — | effectively unresponsive (>2min) |

**Throughput is flat (~80 rps) regardless of concurrency; latency grows linearly.** This is the textbook signature of GIL serialization — extra concurrency only lengthens the queue.

## Findings

1. **`/api/status` is the practical ceiling (~80 rps) and it's the most-polled endpoint.** The UI polls it continuously; it builds a ~26 KB JSON payload under the GIL. Practical capacity ≈ 80 / (polls-per-second-per-client). At one poll / 2s per open tab, ~160 idle tabs saturate the backend before anyone runs a pipeline. An active pipeline run (also CPU, also under the GIL) cuts this much further.
2. **No crash, graceful-ish degradation.** Under overload the server queued and latency collapsed to 12s+, but it did not crash and recovered immediately once load stopped. `ThreadingHTTPServer` + broad `except` handlers keep it alive.
3. **`/workspace` is 423 KB, `no-store`, re-sent on every navigation** — ~30 MB/s at just 70 rps. Bandwidth, not CPU, becomes the limit for page loads.
4. **Good defensive posture at the edge.** Caddy basic auth shields the entire app. In particular the app's `POST /api/auth/login` runs PBKDF2-SHA256 at **260,000 iterations** with **no rate limiting** — a real asymmetric-DoS amplifier — but it is unreachable without basic-auth creds and, on this deployment, app auth is disabled anyway. Caddy (Go) also absorbs slowloris / TLS-handshake floods that would otherwise be dangerous to the unbounded-thread backend.

## Recommendations (priority order)

1. **Cache / cheapen `/api/status`.** Cache the payload for ~1s, or split fast-changing fields from the 26 KB blob, or have the UI poll less aggressively / use conditional requests. Biggest single win.
2. **Put a concurrency/rate limit at Caddy** (`reverse_proxy` + request limits, or `rate_limit`) so a burst can't queue thousands of requests into the GIL-bound backend.
3. **Cap backend threads** — replace bare `ThreadingHTTPServer` with a bounded worker pool (or accept that Caddy limits concurrency for you, per #2).
4. **Make `/workspace` cacheable** (ETag / `Cache-Control` on the static shell) — it's a 423 KB constant served `no-store`.
5. If app-level password is ever enabled, **add rate limiting to `/api/auth/login`** (260k-iteration PBKDF2 with no throttle is a CPU-exhaustion vector).

## What was NOT tested
- Behind-auth pipeline execution under concurrent load (upload → run) — would consume real CPU/GPU under the GIL and is the true worst case; not stress-tested to avoid disrupting a live server.
- Thread-exhaustion / slowloris directly against the backend (Caddy is expected to absorb it; not verified).
