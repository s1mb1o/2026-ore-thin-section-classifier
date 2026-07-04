# Plan 46: Ore Pipeline File Download Streaming

Date: 2026-07-04

## Problem

The v2 ore pipeline HTTP handler serves reports, artifacts, images, and run ZIP files through `send_file()`. The current implementation reads the whole file into memory before writing the response. Large run ZIPs or large preview artifacts can spike server RAM and make the VM unstable.

## Scope

- Replace full-file `read_bytes()` responses with chunked streaming in `apps/ore_pipeline_web.py`.
- Preserve current response headers: content type, content length, no-store cache policy, and optional download filename.
- Keep this change narrow. HTTP range support is useful later, but not required for this fix.
- Add focused regression coverage in `tests/test_ore_pipeline_web.py`.

## Implementation Steps

1. Add a small download chunk-size constant.
2. Update `send_file()` to stat the file, send headers, then write chunks from an open file handle.
3. Tolerate client disconnects during streaming without crashing the server thread.
4. Add a regression test that patches `Path.read_bytes()` to fail and confirms artifact/file responses still work.
5. Run the ore pipeline web test suite and restart the local UI service.

## Acceptance Criteria

- `send_file()` no longer uses `Path.read_bytes()`.
- Artifact/report/ZIP routes keep returning correct `Content-Length` and response bodies.
- Large files are sent using bounded memory proportional to the chunk size.
- Targeted web tests pass.
