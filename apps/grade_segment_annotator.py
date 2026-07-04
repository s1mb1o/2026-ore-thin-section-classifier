#!/usr/bin/env python3
"""Minimal per-segment grade annotator (ordinary vs fine).

Shows one sulfide segment at a time as an expanded context crop with a thin red
contour marking the exact component to be trained on, plus a scale bar. You press
one key/button per segment. Segments are presented largest-area-first (they drive
the % the most). Labels are appended to <dataset>/labels.csv and the tool resumes
from where you left off.

    python3 apps/grade_segment_annotator.py --dataset-dir outputs/grade_segments_v0 --port 8765

Keys:  1 = рядовая (ordinary)   2 = тонкая (fine)   3 = пропустить (skip)
       4 = не руда/артефакт      ←/Backspace = назад (undo)
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

LABELS = {"ordinary": "рядовая", "fine": "тонкая", "skip": "пропуск", "not_ore": "не руда"}

STATE_LOCK = threading.Lock()


class Store:
    def __init__(self, dataset_dir: Path):
        self.dir = dataset_dir
        self.segments = self._load_segments()
        self.order = [s["seg_id"] for s in self.segments]  # already area-desc in csv
        self.by_id = {s["seg_id"]: s for s in self.segments}
        self.labels_path = dataset_dir / "labels.csv"
        self.labels = self._load_labels()

    def _load_segments(self):
        rows = list(csv.DictReader((self.dir / "segments.csv").open(encoding="utf-8")))
        rows.sort(key=lambda r: -int(r["area_px"]))
        return rows

    def _load_labels(self):
        d = {}
        if self.labels_path.exists():
            for r in csv.DictReader(self.labels_path.open(encoding="utf-8")):
                d[r["seg_id"]] = r["label"]
        return d

    def save_label(self, seg_id: str, label: str):
        with STATE_LOCK:
            self.labels[seg_id] = label
            new = not self.labels_path.exists()
            with self.labels_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["seg_id", "label", "image", "mag", "area_px"])
                s = self.by_id[seg_id]
                w.writerow([seg_id, label, s["image"], s["mag"], s["area_px"]])

    def counts(self):
        c = {k: 0 for k in LABELS}
        for v in self.labels.values():
            c[v] = c.get(v, 0) + 1
        return c

    def next_unlabeled(self, start=0):
        for i in range(start, len(self.order)):
            if self.order[i] not in self.labels:
                return i
        return None

    def payload(self, idx):
        if idx is None or idx >= len(self.order):
            return {"done": True, "counts": self.counts(), "total": len(self.order),
                    "labeled": len(self.labels)}
        s = self.by_id[self.order[idx]]
        return {
            "done": False, "index": idx, "total": len(self.order),
            "labeled": len(self.labels), "counts": self.counts(),
            "seg_id": s["seg_id"], "image": s["image"], "mag": s["mag"],
            "area_px": int(s["area_px"]),
            "bbox": f'{s["bbox_w"]}x{s["bbox_h"]}',
            "existing": self.labels.get(s["seg_id"]),
            "preview_url": f'/preview/{s["seg_id"]}',
        }


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Grade annotator</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;overflow:hidden}
 #top{display:flex;gap:16px;align-items:center;padding:8px 14px;background:#1b1b1b;border-bottom:1px solid #333}
 #bar{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden}
 #barfill{height:100%;background:#4caf50;width:0}
 #img{display:flex;align-items:center;justify-content:center;height:calc(100vh - 132px)}
 #img img{max-width:96vw;max-height:100%;object-fit:contain;border:1px solid #333}
 #btns{display:flex;gap:10px;justify-content:center;padding:12px;background:#1b1b1b;border-top:1px solid #333}
 button{font-size:16px;padding:12px 20px;border:0;border-radius:8px;cursor:pointer;color:#fff}
 .b1{background:#2e7d32}.b2{background:#c62828}.b3{background:#555}.b4{background:#37474f}.bu{background:#444}
 .k{opacity:.6;font-size:12px}
 .meta{font-size:13px;color:#aaa}
 .done{font-size:28px;text-align:center;margin-top:20vh}
</style></head><body>
<div id=top>
 <b>Grade annotator</b>
 <div id=bar><div id=barfill></div></div>
 <span class=meta id=meta></span>
 <span class=meta id=cnt></span>
</div>
<div id=img><img id=seg src=""></div>
<div id=btns>
 <button class=b1 onclick="lab('ordinary')">Рядовая <span class=k>1</span></button>
 <button class=b2 onclick="lab('fine')">Тонкая <span class=k>2</span></button>
 <button class=b3 onclick="lab('skip')">Пропуск <span class=k>3</span></button>
 <button class=b4 onclick="lab('not_ore')">Не руда <span class=k>4</span></button>
 <button class=bu onclick="undo()">← Назад <span class=k>Backspace</span></button>
</div>
<script>
let cur=null;
function render(d){
 if(d.done){document.body.innerHTML='<div class=done>Готово ✔<br>'+d.labeled+' / '+d.total+' размечено</div>';return;}
 cur=d;
 document.getElementById('seg').src=d.preview_url+'?t='+Date.now();
 document.getElementById('meta').textContent=d.image+'  ·  '+d.mag+'  ·  '+d.area_px.toLocaleString()+' px  ·  bbox '+d.bbox+(d.existing?('  ·  БЫЛО: '+d.existing):'');
 let c=d.counts;
 document.getElementById('cnt').textContent='ряд '+c.ordinary+' / тонк '+c.fine+' / проп '+c.skip+' / не-руда '+c.not_ore+'  ('+d.labeled+'/'+d.total+')';
 document.getElementById('barfill').style.width=(100*d.labeled/d.total)+'%';
}
async function load(){render(await (await fetch('/api/current')).json());}
async function lab(l){if(!cur)return;render(await (await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seg_id:cur.seg_id,label:l})})).json());}
async function undo(){render(await (await fetch('/api/prev',{method:'POST'})).json());}
document.onkeydown=e=>{
 if(e.key=='1')lab('ordinary');else if(e.key=='2')lab('fine');
 else if(e.key=='3')lab('skip');else if(e.key=='4')lab('not_ore');
 else if(e.key=='Backspace'||e.key=='ArrowLeft'){e.preventDefault();undo();}
};
load();
</script></body></html>"""


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/api/current":
                idx = store.next_unlabeled(0)
                return self._send(200, store.payload(idx))
            if path.startswith("/preview/"):
                seg = path[len("/preview/"):]
                s = store.by_id.get(seg)
                if not s:
                    return self._send(404, {"error": "no seg"})
                data = (store.dir / s["preview"]).read_bytes()
                return self._send(200, data, "image/jpeg")
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or "{}") if length else {}
            if path == "/api/label":
                store.save_label(body["seg_id"], body["label"])
                idx = store.next_unlabeled(0)
                return self._send(200, store.payload(idx))
            if path == "/api/prev":
                # step back to the most recently labeled segment (by order) and reopen it
                last = None
                for i, sid in enumerate(store.order):
                    if sid in store.labels:
                        last = i
                if last is not None:
                    seg = store.order[last]
                    with STATE_LOCK:
                        store.labels.pop(seg, None)
                        # rewrite labels.csv without that seg
                        _rewrite_labels(store)
                    return self._send(200, store.payload(last))
                return self._send(200, store.payload(store.next_unlabeled(0)))
            return self._send(404, {"error": "not found"})

    return H


def _rewrite_labels(store: Store):
    with store.labels_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seg_id", "label", "image", "mag", "area_px"])
        for sid in store.order:
            if sid in store.labels:
                s = store.by_id[sid]
                w.writerow([sid, store.labels[sid], s["image"], s["mag"], s["area_px"]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    store = Store(args.dataset_dir)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Grade segment annotator: http://{args.host}:{args.port}/")
    print(f"segments: {len(store.order)}  already labeled: {len(store.labels)}")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
