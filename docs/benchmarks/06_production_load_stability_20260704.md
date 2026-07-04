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
- Thread-exhaustion / slowloris directly against the backend (Caddy is expected to absorb it; not verified).

---

## CRITICAL — Upload decompression-bomb DoS (verified against production, 2026-07-05)

**Severity: critical.** A single authenticated user can take the entire server offline and trigger an OOM kill with a handful of ~6 MB requests.

### Root cause
- `apps/ore_pipeline_web.py:78` sets `Image.MAX_IMAGE_PIXELS = None`, **disabling PIL's decompression-bomb guard entirely.** (`src/ore_classifier/resident_pipeline.py:51` does the same.)
- The upload path decodes the full image with no dimension check: `image_dimensions()` is lazy (reads only the header), but `_register_upload_file` → `save_preview_pyramid(load_image_pil(...))` forces a full raster decode. For PNG, `image.draft()` is a no-op and `ImageOps.exif_transpose()` (`load_image_pil`, line ~315) loads the full raster *before* the thumbnail downscale.
- `MAX_UPLOAD_BYTES = 2 GiB` bounds the *compressed* bytes only. A 46000×46000 solid-color PNG is **6.17 MB on the wire but decodes to ~6.3 GB of RGB raster** (≈1020:1), plus preview-pyramid copies.
- The multipart handler also amplifies raw bytes: `rfile.read(length)` → concat into `message_bytes` → `BytesParser` → `get_payload(decode=True)` holds ~3–4 copies of the body in RAM per request, thread-per-connection with no cap.

### Reproduction
Generated `bomb.png` (46000×46000 RGB, 6.17 MB file) with streaming zlib. Server baseline: 67.4 GB RAM, 16 vCPU, 65 GB free.

- **Single upload:** occupied a worker thread for **57 s**, consumed ~6.5 GB RAM (avail 65 → 58.5 GB), health → `warning`, returned HTTP 200. No guard rejected it.
- **24 concurrent uploads (~148 MB total on the wire):**
  - `t≈0s` health → `warning`.
  - `t≈8s–44s` **`/api/status` UNREACHABLE** (timeouts) — 16 cores saturated on GIL-releasing libpng decode + memory pressure; even the cheapest endpoint could not respond for ~36 s.
  - `t≈50s` peak captured: **avail RAM 65 GB → 2.5 GB (3.7%), health = `error`.**
  - All 24 client requests returned **HTTP 502** at ~50.8 s — the Python backend was **OOM-killed**.
  - A supervisor **auto-restarted** the process (~`t≈56s`, RAM back to 66 GB, load1m spiked to 17). In-memory state (active runs/jobs) is lost on restart.

**Net: ~148 MB of uploads → full outage for ~40 s + process kill/restart. Trivially repeatable; a loop would keep the service permanently down.**

### Recommendations (priority order)
1. **Restore a pixel cap:** set `Image.MAX_IMAGE_PIXELS` to a value above the largest legitimate thin-section/panorama but far below bomb territory (e.g. ~300–500 MP), and catch `Image.DecompressionBombError` → HTTP 400. Removing the `= None` override is the single highest-impact fix.
2. **Reject by declared dimensions before decode:** `image_dimensions()` already reads W×H cheaply from the header — refuse `width*height` over the cap in `register_upload_from_bytes` *before* any full decode/preview work.
3. **Bound heavy-operation concurrency:** a semaphore limiting simultaneous decode/preprocess/run operations so N uploads queue instead of all allocating rasters at once (also protects CPU/GIL and the `/api/status` responsiveness observed above).
4. **Lower `MAX_UPLOAD_BYTES`** to a realistic ceiling and set a Caddy `request_body max_size` at the edge.
5. **Stream the multipart body to a temp file** instead of `rfile.read(length)` + in-memory MIME parse, to remove the 3–4× raw-byte amplification.

### Cleanup note
The flood wrote up to ~24 orphaned `bomb.png` upload directories under `/data/ore_pipeline_ui/uploads/` (≈6 MB each; the decode crashed before registration completed, so they are unregistered orphans) plus one fully-registered test upload. Worth pruning server-side.

## What was NOT tested
- Behind-auth pipeline execution under concurrent load (upload → run) — would consume real CPU/GPU under the GIL and is the true worst case; not stress-tested to avoid disrupting a live server.
