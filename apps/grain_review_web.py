#!/usr/bin/env python3
"""Grain review / labeling app (path B, stage 2).

A local browser app to human-classify sulfide grains as ordinary vs fine
intergrowth (or 'uncertain'). It shows a paginated grid of grain crops produced
by `scripts/build_grain_dataset.py`, pre-labelled by the heuristic, and persists
human corrections to `annotations.json` in the dataset directory, keyed by
`grain_uid`. Those annotations feed `train_grain_classifier.py` /
`aggregate_grade_from_grains.py`.

Stdlib only (http.server), no framework — same architecture as
`apps/talc_review_web.py`. Keyboard: O=ordinary, F=fine, U=uncertain, arrows to
move. Run:

    python3 apps/grain_review_web.py --dataset-dir outputs/grain_dataset_v0 --port 0
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

GRAIN_CLASSES = ["ordinary_intergrowth", "fine_intergrowth", "uncertain"]
MAX_POST_BYTES = 8 * 1024 * 1024

# Full morphology feature set carried per grain in grains_manifest.csv, surfaced
# in the labeling UI so the annotator can decide ordinary vs fine from the same
# numbers the v2 pipeline reports.
FEATURE_FIELDS = [
    "area_px",
    "footprint_area_px",
    "dark_inside_area_px",
    "dark_inside_ratio",
    "solidity",
    "compactness",
    "boundary_complexity",
    "bbox_w",
    "bbox_h",
]

# Heuristic "fine" thresholds — mirror ComponentRuleConfig defaults
# (src/ore_classifier/component_analysis.py). A grain is pre-labelled fine if ANY
# of these trip. Shown to the annotator as the reason behind the pre-label.
FINE_DARK_INSIDE_RATIO = 0.18
FINE_SOLIDITY_MAX = 0.62
FINE_COMPACTNESS_MAX = 0.12


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiError(RuntimeError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class GrainReviewStore:
    """Shared, thread-safe state: the grain manifest + the annotations file."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.crops_root = (self.dataset_dir / "crops").resolve()
        self.manifest_path = self.dataset_dir / "grains_manifest.csv"
        self.annotations_path = self.dataset_dir / "annotations.json"
        if not self.manifest_path.exists():
            raise SystemExit(f"grains_manifest.csv not found in {self.dataset_dir}")
        self.lock = threading.RLock()
        self.grains: list[dict[str, str]] = self._load_manifest()
        self.index_by_uid = {g["grain_uid"]: i for i, g in enumerate(self.grains)}
        self.labels: dict[str, dict[str, Any]] = self._load_annotations()

    def _load_manifest(self) -> list[dict[str, str]]:
        with self.manifest_path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _load_annotations(self) -> dict[str, dict[str, Any]]:
        if not self.annotations_path.exists():
            return {}
        payload = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        labels = payload.get("labels", {}) if isinstance(payload, dict) else {}
        return {k: v for k, v in labels.items() if isinstance(v, dict)}

    def _save_annotations(self) -> None:
        payload = {
            "schema_version": "grain-annotations-v0.1",
            "updated_at": utc_now_iso(),
            "dataset_dir": str(self.dataset_dir),
            "labels": self.labels,
        }
        tmp = self.annotations_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.annotations_path)

    def stats(self) -> dict[str, Any]:
        counts = {cls: 0 for cls in GRAIN_CLASSES}
        for entry in self.labels.values():
            label = entry.get("label")
            if label in counts:
                counts[label] += 1
        return {"total": len(self.grains), "labeled": len(self.labels), "counts": counts}

    def page(self, *, offset: int, limit: int, grade: str, view: str) -> dict[str, Any]:
        with self.lock:
            filtered = self._filter(grade=grade, view=view)
            window = filtered[offset : offset + limit]
            items = [self._item_payload(i) for i in window]
            return {
                "items": items,
                "offset": offset,
                "limit": limit,
                "filtered_total": len(filtered),
                "stats": self.stats(),
            }

    def _filter(self, *, grade: str, view: str) -> list[int]:
        result = []
        for i, g in enumerate(self.grains):
            if grade not in ("", "all") and g.get("grade_label") != grade:
                continue
            labeled = g["grain_uid"] in self.labels
            if view == "unlabeled" and labeled:
                continue
            if view == "labeled" and not labeled:
                continue
            result.append(i)
        return result

    def _item_payload(self, index: int) -> dict[str, Any]:
        g = self.grains[index]
        uid = g["grain_uid"]
        features = {key: _num(g.get(key)) for key in FEATURE_FIELDS}
        dir_ = features.get("dark_inside_ratio")
        sol = features.get("solidity")
        cmp_ = features.get("compactness")
        fine_signals = {
            "dark_inside_ratio": dir_ is not None and dir_ >= FINE_DARK_INSIDE_RATIO,
            "solidity": sol is not None and sol <= FINE_SOLIDITY_MAX,
            "compactness": cmp_ is not None and cmp_ <= FINE_COMPACTNESS_MAX,
        }
        reasons: list[str] = []
        if fine_signals["dark_inside_ratio"]:
            reasons.append(f"тёмное внутри {dir_:.2f} ≥ {FINE_DARK_INSIDE_RATIO}")
        if fine_signals["solidity"]:
            reasons.append(f"выпуклость {sol:.2f} ≤ {FINE_SOLIDITY_MAX}")
        if fine_signals["compactness"]:
            reasons.append(f"компактность {cmp_:.3f} ≤ {FINE_COMPACTNESS_MAX}")
        return {
            "grain_uid": uid,
            "crop_url": "/crops/" + quote(g["crop_path"].split("crops/", 1)[-1]),
            "grade_label": g.get("grade_label", ""),
            "heuristic_label": g.get("heuristic_label", ""),
            "features": features,
            "fine_signals": fine_signals,
            "fine_reasons": reasons,
            "label": self.labels.get(uid, {}).get("label"),
        }

    def annotate(self, grain_uid: str, label: str | None) -> dict[str, Any]:
        if grain_uid not in self.index_by_uid:
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown grain_uid {grain_uid}")
        if label is not None and label not in GRAIN_CLASSES:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"bad label {label}")
        with self.lock:
            if label is None:
                self.labels.pop(grain_uid, None)
            else:
                self.labels[grain_uid] = {"label": label, "at": utc_now_iso()}
            self._save_annotations()
            return self.stats()

    def crop_file(self, rel: str) -> Path:
        candidate = (self.crops_root / rel).resolve()
        # is_relative_to enforces a real path-boundary; a bare startswith would let
        # a sibling dir sharing the 'crops' name prefix (crops_backup/, crops2/…) escape.
        if not candidate.is_relative_to(self.crops_root) or not candidate.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "crop not found")
        return candidate


class GrainReviewHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: GrainReviewStore) -> None:
        super().__init__(server_address, GrainReviewHandler)
        self.store = store


class GrainReviewHandler(BaseHTTPRequestHandler):
    server: "GrainReviewHTTPServer"

    def log_message(self, *args: Any) -> None:  # keep the console quiet
        return

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except ApiError as exc:
            self.send_json({"error": exc.message}, exc.status)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except ApiError as exc:
            self.send_json({"error": exc.message}, exc.status)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_html(render_page())
            return
        if path == "/api/page":
            q = parse_qs(parsed.query)
            payload = self.server.store.page(
                offset=max(0, int((q.get("offset", ["0"])[0]) or 0)),
                limit=min(200, max(1, int((q.get("limit", ["60"])[0]) or 60))),
                grade=(q.get("grade", ["all"])[0]),
                view=(q.get("view", ["all"])[0]),
            )
            self.send_json(payload)
            return
        if path.startswith("/crops/"):
            rel = unquote(path[len("/crops/") :])
            self.send_file(self.server.store.crop_file(rel))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def _handle_post(self) -> None:
        parsed = urlparse(self.path)
        payload = self.read_json_payload()
        if parsed.path == "/api/annotate":
            stats = self.server.store.annotate(str(payload.get("grain_uid", "")), payload.get("label"))
            self.send_json({"ok": True, "stats": stats})
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > MAX_POST_BYTES:
            raise ApiError(HTTPStatus.BAD_REQUEST, "empty or oversized body")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid json: {exc}") from exc
        if not isinstance(data, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "body must be an object")
        return data

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, markup: str) -> None:
        body = markup.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        import mimetypes

        data = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def render_page() -> str:
    return """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка зёрен — ordinary / fine</title>
<style>
:root{color-scheme:dark;--bg:#12151b;--panel:#1b1f28;--line:#2b313d;--text:#e7ecf3;--muted:#93a0b4;
--ord:#1fa25a;--fine:#d83f45;--unc:#8a93a5;--accent:#3aa0a4}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--text)}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
header h1{font-size:15px;margin:0 12px 0 0;font-weight:650}
select,button{background:#232833;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:13px}
button{cursor:pointer}
.prog{color:var(--muted);font-size:13px}
.wrap{display:flex;align-items:flex-start}
main{flex:1;min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;padding:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column;cursor:pointer}
.card.sel{outline:2px solid var(--accent)}
.card img{width:100%;height:120px;object-fit:contain;background:#0c0e12}
.meta{font-size:10px;color:var(--muted);padding:4px 6px 0;display:flex;justify-content:space-between;align-items:center}
.chips{display:flex;gap:3px;padding:3px 6px 4px;flex-wrap:wrap}
.mc{font-size:10px;padding:1px 4px;border-radius:5px;background:#232833;color:var(--muted);font-variant-numeric:tabular-nums}
.mc.fs{background:rgba(216,63,69,.22);color:#f2969a}
.badge{font-size:10px;padding:1px 5px;border-radius:6px}
.b-ord{background:rgba(31,162,90,.2);color:#7fe0a8}.b-fine{background:rgba(216,63,69,.2);color:#f2969a}
.row{display:flex;margin-top:auto}.row button{flex:1;border-radius:0;border:0;border-top:1px solid var(--line);font-size:12px;padding:5px 0}
.row .a-ord.on{background:var(--ord);color:#04170c}.row .a-fine.on{background:var(--fine);color:#1c0405}.row .a-unc.on{background:var(--unc);color:#10131a}
aside{width:320px;flex:none;position:sticky;top:52px;height:calc(100vh - 52px);overflow:auto;border-left:1px solid var(--line);padding:14px;background:var(--panel)}
aside h2{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:600}
aside img{width:100%;max-height:300px;object-fit:contain;background:#0c0e12;border-radius:8px}
.verdict{font-size:13px;margin:10px 0;line-height:1.4}
.ftab{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.ftab td{padding:3px 4px;border-bottom:1px solid var(--line)}
.ftab td:last-child{text-align:right}
.ftab tr.fs td{color:#f2969a}
.dbtns{display:flex;gap:6px;margin-top:12px}
.dbtns button{flex:1}
.dbtns .on.a-ord{background:var(--ord);color:#04170c;border-color:var(--ord)}
.dbtns .on.a-fine{background:var(--fine);color:#1c0405;border-color:var(--fine)}
.dbtns .on.a-unc{background:var(--unc);color:#10131a;border-color:var(--unc)}
.dhint{color:var(--muted);font-size:13px}
.hint{color:var(--muted);font-size:12px;padding:0 16px 20px}
.pager{display:flex;gap:8px;align-items:center;padding:0 16px 20px}
</style></head><body>
<header>
<h1>Разметка зёрен</h1>
<label>Сорт <select id="grade"><option value="all">все</option><option value="ordinary_intergrowth">рядовая</option><option value="fine_intergrowth">труднообог.</option><option value="talcose">оталькован.</option></select></label>
<label>Показ <select id="view"><option value="all">все</option><option value="unlabeled">без метки</option><option value="labeled">размечены</option></select></label>
<span class="prog" id="prog"></span>
<span style="flex:1"></span>
<span class="prog">O=рядовое · F=тонкое · U=неясно · ←/→ навигация</span>
</header>
<div class="pager"><button id="prev">← стр.</button><span class="prog" id="pageinfo"></span><button id="next">стр. →</button></div>
<div class="wrap">
<main>
<div class="grid" id="grid"></div>
<div class="hint">Клик по карточке — выделить; справа — полный отчёт по зерну (те же признаки, что в v2-пайплайне) и подсказка эвристики. Красным подсвечены признаки, по которым срабатывает «тонкое». Кнопки/клавиши присваивают класс, автосохранение в annotations.json.</div>
</main>
<aside id="detail"><div class="dhint">Выберите зерно, чтобы увидеть отчёт по признакам.</div></aside>
</div>
<script>
const state={offset:0,limit:60,grade:'all',view:'all',items:[],sel:0};
const $=s=>document.querySelector(s);
const CLASS_RU={ordinary_intergrowth:'рядовое',fine_intergrowth:'тонкое',uncertain:'неясно'};
function fmt(x,d){const n=parseFloat(x);return (x===null||isNaN(n))?'—':n.toFixed(d===undefined?2:d);}
function statsText(s){return `размечено ${s.labeled}/${s.total} · рядовых ${s.counts.ordinary_intergrowth} · тонких ${s.counts.fine_intergrowth} · неясно ${s.counts.uncertain}`;}
async function load(){
  const q=new URLSearchParams({offset:state.offset,limit:state.limit,grade:state.grade,view:state.view});
  const r=await fetch('/api/page?'+q);const d=await r.json();
  state.items=d.items;state.sel=0;
  $('#prog').textContent=statsText(d.stats);
  $('#pageinfo').textContent=`${d.filtered_total?d.offset+1:0}–${Math.min(d.offset+d.limit,d.filtered_total)} из ${d.filtered_total}`;
  render();
}
function chip(name,val,isFine,d){return `<span class="mc ${isFine?'fs':''}">${name} ${fmt(val,d)}</span>`;}
function render(){
  const g=$('#grid');g.innerHTML='';
  state.items.forEach((it,i)=>{
    const c=document.createElement('div');c.className='card'+(i===state.sel?' sel':'');
    const f=it.features,sg=it.fine_signals;
    const badge=it.heuristic_label==='fine_intergrowth'?'<span class="badge b-fine">эвр: тонкое</span>':'<span class="badge b-ord">эвр: рядовое</span>';
    c.innerHTML=`<img loading="lazy" src="${it.crop_url}">
    <div class="meta">${badge}<span>a=${fmt(f.area_px,0)}</span></div>
    <div class="chips">${chip('d',f.dark_inside_ratio,sg.dark_inside_ratio,2)}${chip('s',f.solidity,sg.solidity,2)}${chip('c',f.compactness,sg.compactness,3)}</div>
    <div class="row">
      <button class="a-ord ${it.label==='ordinary_intergrowth'?'on':''}" data-l="ordinary_intergrowth">рядовое</button>
      <button class="a-fine ${it.label==='fine_intergrowth'?'on':''}" data-l="fine_intergrowth">тонкое</button>
      <button class="a-unc ${it.label==='uncertain'?'on':''}" data-l="uncertain">?</button>
    </div>`;
    c.onclick=(e)=>{if(e.target.tagName!=='BUTTON'){state.sel=i;highlight();}};
    c.querySelectorAll('button').forEach(b=>b.onclick=()=>{state.sel=i;assign(b.dataset.l);});
    g.appendChild(c);
  });
  renderDetail();
}
function highlight(){document.querySelectorAll('.card').forEach((c,i)=>c.classList.toggle('sel',i===state.sel));renderDetail();}
function frow(label,val,d,isFine){return `<tr class="${isFine?'fs':''}"><td>${label}</td><td>${fmt(val,d)}</td></tr>`;}
function renderDetail(){
  const a=$('#detail');const it=state.items[state.sel];
  if(!it){a.innerHTML='<div class="dhint">Выберите зерно, чтобы увидеть отчёт по признакам.</div>';return;}
  const f=it.features,sg=it.fine_signals;
  const verdict=it.heuristic_label==='fine_intergrowth'
    ? `<b style="color:var(--fine)">тонкое</b> — сработало: ${it.fine_reasons.join('; ')}`
    : '<b style="color:var(--ord)">рядовое</b> — ни один порог «тонкого» не сработал';
  const cur=it.label?`ваша метка: <b>${CLASS_RU[it.label]}</b>`:'ещё не размечено';
  a.innerHTML=`<h2>Отчёт по зерну · ${it.grade_label}</h2>
  <img src="${it.crop_url}">
  <div class="verdict">Эвристика: ${verdict}<br><span class="dhint">${cur}</span></div>
  <table class="ftab">
    ${frow('Доля тёмного (замещение)',f.dark_inside_ratio,2,sg.dark_inside_ratio)}
    ${frow('Выпуклость (solidity)',f.solidity,2,sg.solidity)}
    ${frow('Компактность',f.compactness,3,sg.compactness)}
    ${frow('Сложность границы',f.boundary_complexity,2,false)}
    ${frow('Площадь, px',f.area_px,0,false)}
    ${frow('Площадь контура, px',f.footprint_area_px,0,false)}
    ${frow('Тёмное внутри, px',f.dark_inside_area_px,0,false)}
    <tr><td>BBox</td><td>${fmt(f.bbox_w,0)} × ${fmt(f.bbox_h,0)}</td></tr>
  </table>
  <div class="dbtns">
    <button class="a-ord ${it.label==='ordinary_intergrowth'?'on':''}" data-l="ordinary_intergrowth">рядовое (O)</button>
    <button class="a-fine ${it.label==='fine_intergrowth'?'on':''}" data-l="fine_intergrowth">тонкое (F)</button>
    <button class="a-unc ${it.label==='uncertain'?'on':''}" data-l="uncertain">? (U)</button>
  </div>`;
  a.querySelectorAll('.dbtns button').forEach(b=>b.onclick=()=>assign(b.dataset.l));
}
async function assign(label){
  const it=state.items[state.sel];if(!it)return;
  const newLabel=it.label===label?null:label;
  await fetch('/api/annotate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({grain_uid:it.grain_uid,label:newLabel})});
  it.label=newLabel;
  if(newLabel&&state.sel<state.items.length-1){state.sel++;}
  render();
  load_stats_only();
}
async function load_stats_only(){const r=await fetch('/api/page?'+new URLSearchParams({offset:0,limit:1,grade:state.grade,view:state.view}));const d=await r.json();$('#prog').textContent=statsText(d.stats);}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT')return;
  if(e.key==='ArrowRight'){state.sel=Math.min(state.sel+1,state.items.length-1);highlight();}
  else if(e.key==='ArrowLeft'){state.sel=Math.max(state.sel-1,0);highlight();}
  else if(e.key.toLowerCase()==='o'){assign('ordinary_intergrowth');}
  else if(e.key.toLowerCase()==='f'){assign('fine_intergrowth');}
  else if(e.key.toLowerCase()==='u'){assign('uncertain');}
});
$('#grade').onchange=e=>{state.grade=e.target.value;state.offset=0;load();};
$('#view').onchange=e=>{state.view=e.target.value;state.offset=0;load();};
$('#prev').onclick=()=>{state.offset=Math.max(0,state.offset-state.limit);load();};
$('#next').onclick=()=>{state.offset+=state.limit;load();};
load();
</script></body></html>"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grain review / labeling app.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Output dir of build_grain_dataset.py (grains_manifest.csv + crops/).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 asks the OS for a free port.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    store = GrainReviewStore(args.dataset_dir)
    server = GrainReviewHTTPServer((args.host, args.port), store)
    host, port = server.server_address[0], server.server_address[1]
    stats = store.stats()
    print(f"Grain review: http://{host}:{port}/  ({stats['labeled']}/{stats['total']} labelled)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
