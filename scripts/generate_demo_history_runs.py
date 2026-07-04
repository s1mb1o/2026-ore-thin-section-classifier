#!/usr/bin/env python3
"""Drive the ore pipeline web app REST API to pre-generate demo runs into /history.

Usage:
    python gen_demo_runs.py <base_url> <label> <image_path> [<image_path> ...]

Prints a JSON line per run: {"label","image","run_id","status","seconds"}.
"""
import json
import mimetypes
import sys
import time
import urllib.request
import uuid
from pathlib import Path

TERMINAL_OK = {"complete", "completed", "done"}
TERMINAL_BAD = {"failed", "error", "cancelled"}


def _post_multipart(url, field_name, filepath):
    boundary = "----demo" + uuid.uuid4().hex
    fname = Path(filepath).name
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    with open(filepath, "rb") as fh:
        data = fh.read()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{fname}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        data,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def _post_json(url, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def _get_json(url):
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode())


def run_one(base, label, image_path, poll_timeout=1800):
    t0 = time.monotonic()
    up = _post_multipart(base + "/api/uploads", "file", image_path)
    upload_id = up["upload_id"]
    started = _post_json(base + "/api/runs/start", {"upload_id": upload_id})
    run_id = started.get("run_id") or started.get("id")
    status = started.get("status", "queued")
    while status not in TERMINAL_OK and status not in TERMINAL_BAD:
        if time.monotonic() - t0 > poll_timeout:
            status = "timeout"
            break
        time.sleep(3)
        try:
            run = _get_json(base + f"/api/runs/{run_id}")
        except Exception as exc:  # noqa: BLE001
            status = f"poll-error:{exc}"
            break
        status = run.get("status", status)
    return {
        "label": label,
        "image": Path(image_path).name,
        "run_id": run_id,
        "status": status,
        "seconds": round(time.monotonic() - t0, 1),
    }


def main():
    base = sys.argv[1].rstrip("/")
    label = sys.argv[2]
    images = sys.argv[3:]
    for img in images:
        result = run_one(base, label, img)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
