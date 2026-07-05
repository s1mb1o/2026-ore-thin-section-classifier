#!/usr/bin/env python3
"""Frame-level grade mask review/editor.

Loads each frame with the pipeline's per-pixel class mask (0=not-ore, 1=ordinary,
2=fine) and lets you correct it:
  - recolor the whole ore mask to one class (frame is uniformly ordinary/fine),
  - erasing lasso  -> sets region to not-ore (0),
  - class lasso    -> paints region as ordinary (1) or fine (2),
  - opacity slider for the overlay.
Saves the corrected mask to <out-dir>/<frame_id>/corrected_mask.png plus meta.json
with the recomputed area-based class. Later, crops for training are cut from these.

    python3 apps/grade_mask_review.py \
        --runs-dir outputs/evaluations/ch1_dark_green_notalc_20260704/run/runs \
        --out-dir outputs/grade_mask_review_v0 --port 8766
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
LOCK = threading.Lock()


class Store:
    def __init__(self, runs_dir: Path, out_dir: Path, seed_dir: Path | None = None):
        self.out_dir = out_dir
        self.seed_dir = seed_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.frames = self._scan(runs_dir)
        self.by_id = {f["id"]: f for f in self.frames}

    def _scan(self, runs_dir: Path):
        frames = []
        for ps_path in sorted(glob.glob(str(runs_dir / "**/pipeline_summary.json"), recursive=True)):
            ps = json.loads(Path(ps_path).read_text(encoding="utf-8"))
            run_dir = Path(ps_path).parent
            img = Path(ps["image"])
            mask = run_dir / "ore_analysis" / "intergrowth_mask.png"
            if not img.exists() or not mask.exists():
                continue
            ore = run_dir / "ore_analysis" / "ore_summary.json"
            pred = "?"
            if ore.exists():
                pred = json.loads(ore.read_text(encoding="utf-8")).get("ore_class", "?")
            p = str(img)
            gt = "row_ore" if "Рядовые" in p else ("hard_to_process_ore" if "Труднообогат" in p else "?")
            fid = run_dir.name
            seed = self.seed_dir / f"{fid}.png" if self.seed_dir else None
            init_mask = seed if (seed and seed.exists()) else mask
            folder = f"{img.parent.parent.name}/{img.parent.name}"  # e.g. "…ч1/Рядовые руды"
            frames.append({"id": fid, "image": img, "name": img.name, "mask": init_mask,
                           "rule_mask": mask, "pipeline_class": pred, "gt": gt, "folder": folder,
                           "seeded": bool(seed and seed.exists())})
        return frames

    def corrected_path(self, fid: str) -> Path:
        return self.out_dir / fid / "corrected_mask.png"

    def status(self):
        out = []
        for f in self.frames:
            out.append({"id": f["id"], "name": f["name"], "gt": f["gt"], "folder": f["folder"],
                        "pipeline_class": f["pipeline_class"],
                        "reviewed": self.corrected_path(f["id"]).exists()})
        return out

    def save(self, fid: str, mask_png_bytes: bytes):
        f = self.by_id[fid]
        cls = np.array(Image.open(io.BytesIO(mask_png_bytes)).convert("L"))
        # values arrive as 0/1/2 (R channel already collapsed to L)
        cls = np.clip(cls, 0, 2).astype(np.uint8)
        # match native mask size
        native = np.asarray(Image.open(f["mask"]).convert("L"))
        if cls.shape != native.shape:
            cls = np.array(Image.fromarray(cls).resize((native.shape[1], native.shape[0]), Image.NEAREST))
        d = self.out_dir / fid
        d.mkdir(parents=True, exist_ok=True)
        with LOCK:
            Image.fromarray(cls).save(self.corrected_path(fid))
            ord_area = int((cls == 1).sum()); fine_area = int((cls == 2).sum())
            klass = "row_ore" if ord_area >= fine_area else "hard_to_process_ore"
            if ord_area == 0 and fine_area == 0:
                klass = "barren"
            (d / "meta.json").write_text(json.dumps({
                "frame_id": fid, "name": f["name"], "gt": f["gt"],
                "pipeline_class": f["pipeline_class"], "corrected_class": klass,
                "ordinary_px": ord_area, "fine_px": fine_area,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        return klass


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _img_bytes(self, path: Path, mode: str, fmt: str, max_side=None):
            im = Image.open(path).convert(mode)
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.NEAREST if mode == "L" else Image.BILINEAR)
            buf = io.BytesIO(); im.save(buf, fmt); return buf.getvalue()

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/api/frames":
                return self._send(200, store.status())
            if path.startswith("/img/"):
                f = store.by_id.get(path[5:])
                if not f: return self._send(404, {"error": "no frame"})
                return self._send(200, self._img_bytes(f["image"], "RGB", "JPEG"), "image/jpeg")
            if path.startswith("/mask/"):
                fid = path[6:]; f = store.by_id.get(fid)
                if not f: return self._send(404, {"error": "no frame"})
                raw = "raw=1" in (urlparse(self.path).query or "")
                src = f["mask"] if raw else (store.corrected_path(fid) if store.corrected_path(fid).exists() else f["mask"])
                # remap class values 0/1/2 -> 0/128/255 so the browser decodes them robustly
                m = np.asarray(Image.open(src).convert("L"))
                spread = np.zeros_like(m); spread[m == 1] = 128; spread[m == 2] = 255
                buf = io.BytesIO(); Image.fromarray(spread).save(buf, "PNG")
                return self._send(200, buf.getvalue(), "image/png")
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            if path.startswith("/api/save/"):
                fid = path[len("/api/save/"):]
                klass = store.save(fid, raw)
                return self._send(200, {"ok": True, "corrected_class": klass})
            return self._send(404, {"error": "not found"})

    return H


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>Grade mask review</title>
<style>
 html,body{margin:0;height:100%;background:#0e0e0e;color:#eee;font-family:system-ui,sans-serif}
 #top{display:flex;gap:10px;align-items:center;padding:6px 10px;background:#1b1b1b;border-bottom:1px solid #333;flex-wrap:wrap}
 #wrap{position:relative;height:calc(100vh - 96px);overflow:auto;display:flex;align-items:center;justify-content:center;background:#000}
 canvas{max-width:98vw;max-height:100%;cursor:crosshair;touch-action:none}
 button{font-size:14px;padding:8px 12px;border:0;border-radius:6px;cursor:pointer;color:#fff;background:#444}
 button.on{outline:2px solid #fff}
 .erase{background:#37474f}.ord{background:#1e9955}.fine{background:#c62828}.save{background:#00695c}
 .meta{font-size:12px;color:#aaa}
 input[type=range]{width:120px;vertical-align:middle}
 #list{max-height:100%;overflow:auto}
</style></head><body>
<div id=top>
 <b>Grade mask review</b><span class=meta style="color:#4fc3f7">v4 · 🟩рядовая 🟥тонкая (как web app)</span>
 <span class=meta id=meta></span>
 <span>|</span>
 <button class=ord id=tOrd onclick="setTool('ord')">Лассо: Рядовая <small>r</small></button>
 <button class=fine id=tFine onclick="setTool('fine')">Лассо: Тонкая <small>f</small></button>
 <button class=erase id=tErase onclick="setTool('erase')">Лассо: Стереть <small>e</small></button>
 <span>|</span>
 <button onclick="recolorAll(1)">Всё→Рядовая <small>1</small></button>
 <button onclick="recolorAll(2)">Всё→Тонкая <small>2</small></button>
 <button class=erase onclick="eraseAll()">Стереть всё <small>0</small></button>
 <button onclick="resetMask()">Сброс к пайплайну</button>
 <button id=tHide onclick="toggleOverlay()">Скрыть оверлей <small>пробел</small></button>
 <span>| прозрачность <input type=range id=op min=0 max=100 value=100 oninput="renderComposite();render()"></span>
 <span class=meta id=stats></span>
 <span>|</span>
 <button onclick="prev()">◀ <small>←</small></button>
 <button class=save onclick="save()">Сохранить+далее <small>s</small></button>
 <button onclick="next()">▶ <small>→</small></button>
 <span class=meta id=prog></span>
</div>
<div id=wrap><canvas id=cv></canvas></div>
<script>
let frames=[], fi=0, W=0, H=0, cls=null, tool='ord', drawing=false, pts=[], hideOverlay=false;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const base=document.createElement('canvas'), bctx=base.getContext('2d');
const comp=document.createElement('canvas'), cmp=comp.getContext('2d');
const COL={1:[30,185,85],2:[230,65,65]};   // == web app CLASS_COLORS (ordinary green, fine red)
const ALPHA={1:150,2:160};                  // == web app per-class alpha (of 255)
async function loadList(){frames=await (await fetch('/api/frames')).json();
  fi=frames.findIndex(f=>!f.reviewed); if(fi<0)fi=0; await loadFrame();}
function setTool(t){tool=t;for(const[id,tt]of[['tOrd','ord'],['tFine','fine'],['tErase','erase']])
  document.getElementById(id).classList.toggle('on',tt===t);}
async function loadFrame(){
  const f=frames[fi];
  const img=new Image(); img.src='/img/'+f.id+'?t='+Date.now();
  await img.decode();
  W=img.naturalWidth; H=img.naturalHeight;
  cv.width=W; cv.height=H; base.width=W; base.height=H; comp.width=W; comp.height=H;
  bctx.drawImage(img,0,0);
  // load class mask
  const m=new Image(); m.src='/mask/'+f.id+'?t='+Date.now(); await m.decode();
  const tmp=document.createElement('canvas'); tmp.width=W; tmp.height=H;
  const tc=tmp.getContext('2d'); tc.drawImage(m,0,0,W,H);
  const md=tc.getImageData(0,0,W,H).data;
  cls=new Uint8Array(W*H);
  for(let i=0;i<W*H;i++){let v=md[i*4]; cls[i]=v>=192?2:(v>=64?1:0);} // robust: 0/128/255
  document.getElementById('meta').textContent=f.name+'  ·  📁 '+f.folder+'  ·  GT: '+ru(f.gt)+'  ·  пайплайн: '+ru(f.pipeline_class)+(f.reviewed?'  ·  ✔ проверен':'');
  document.getElementById('prog').textContent=(fi+1)+'/'+frames.length+'  ('+frames.filter(x=>x.reviewed).length+' готово)';
  setTool(tool); updateStats(); renderComposite(); render();
}
function updateStats(){let b=0,o=0,f=0;for(let i=0;i<W*H;i++){const c=cls[i];if(c==1)o++;else if(c==2)f++;else b++;}
  const t=W*H;document.getElementById('stats').textContent='фон '+(100*b/t).toFixed(0)+'% · рядовая '+(100*o/t).toFixed(0)+'% · тонкая '+(100*f/t).toFixed(0)+'%';}
function ru(c){return c=='row_ore'?'рядовая':c=='hard_to_process_ore'?'тонкая':c=='barren'?'пусто':c;}
function renderComposite(){ // heavy: base + class overlay -> comp (call only when cls/opacity change)
  cmp.drawImage(base,0,0);
  const op=document.getElementById('op').value/100;
  if(op>0 && !hideOverlay){
    const ov=cmp.getImageData(0,0,W,H), d=ov.data;
    for(let i=0;i<W*H;i++){const c=cls[i]; if(c){const col=COL[c], a=op*ALPHA[c]/255; // flat RGBA over image, like colored_overlay()
      d[i*4]=d[i*4]*(1-a)+col[0]*a; d[i*4+1]=d[i*4+1]*(1-a)+col[1]*a; d[i*4+2]=d[i*4+2]*(1-a)+col[2]*a;}}
    cmp.putImageData(ov,0,0);
  }
}
function render(){ // cheap: blit comp + in-progress lasso
  ctx.drawImage(comp,0,0);
  if(drawing&&pts.length>1){ctx.strokeStyle='#fff';ctx.lineWidth=Math.max(1,W/900);ctx.beginPath();
    ctx.moveTo(pts[0].x,pts[0].y);for(const p of pts)ctx.lineTo(p.x,p.y);ctx.stroke();}
}
function toXY(e){const r=cv.getBoundingClientRect();
  return {x:(e.clientX-r.left)*(W/r.width), y:(e.clientY-r.top)*(H/r.height)};}
cv.onpointerdown=e=>{drawing=true;pts=[toXY(e)];cv.setPointerCapture(e.pointerId);};
cv.onpointermove=e=>{if(drawing){pts.push(toXY(e));render();}};
cv.onpointerup=e=>{if(!drawing)return;drawing=false;if(pts.length>2)fillPoly(pts,tool);pts=[];renderComposite();render();updateStats();};
function fillPoly(poly,t){
  const val=t=='erase'?0:(t=='ord'?1:2);
  let minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9;
  for(const p of poly){minx=Math.min(minx,p.x);miny=Math.min(miny,p.y);maxx=Math.max(maxx,p.x);maxy=Math.max(maxy,p.y);}
  minx=Math.max(0,Math.floor(minx));miny=Math.max(0,Math.floor(miny));
  maxx=Math.min(W-1,Math.ceil(maxx));maxy=Math.min(H-1,Math.ceil(maxy));
  // rasterize polygon in its bbox on a temp canvas
  const bw=maxx-minx+1, bh=maxy-miny+1; if(bw<=0||bh<=0)return;
  const t2=document.createElement('canvas'); t2.width=bw; t2.height=bh;
  const c2=t2.getContext('2d'); c2.beginPath(); c2.moveTo(poly[0].x-minx,poly[0].y-miny);
  for(const p of poly)c2.lineTo(p.x-minx,p.y-miny); c2.closePath(); c2.fillStyle='#fff'; c2.fill();
  const pd=c2.getImageData(0,0,bw,bh).data;
  for(let y=0;y<bh;y++)for(let x=0;x<bw;x++){ if(pd[(y*bw+x)*4+3]>10){
    const gi=(miny+y)*W+(minx+x);
    // class lasso reclassifies ONLY ore pixels in the region (never touches gangue/фон);
    // erase lasso removes ore. The pipeline's ore footprint is otherwise untouched.
    if(val==0)cls[gi]=0; else if(cls[gi]!=0)cls[gi]=val; }}
}
function recolorAll(v){for(let i=0;i<W*H;i++)if(cls[i]!=0)cls[i]=v;renderComposite();render();updateStats();} // only ore pixels
function eraseAll(){cls.fill(0);renderComposite();render();updateStats();}
function toggleOverlay(){hideOverlay=!hideOverlay;document.getElementById('tHide').classList.toggle('on',hideOverlay);renderComposite();render();}
async function resetMask(){const f=frames[fi];
  const m=new Image(); m.src='/mask/'+f.id+'?raw=1&t='+Date.now(); await m.decode();
  const tmp=document.createElement('canvas'); tmp.width=W; tmp.height=H;
  const tc=tmp.getContext('2d'); tc.drawImage(m,0,0,W,H);
  const md=tc.getImageData(0,0,W,H).data;
  for(let i=0;i<W*H;i++){let v=md[i*4]; cls[i]=v>=192?2:(v>=64?1:0);}
  renderComposite(); render();}
function classMaskPNG(){const t=document.createElement('canvas');t.width=W;t.height=H;
  const c=t.getContext('2d');const im=c.createImageData(W,H);
  for(let i=0;i<W*H;i++){im.data[i*4]=cls[i];im.data[i*4+1]=cls[i];im.data[i*4+2]=cls[i];im.data[i*4+3]=255;}
  c.putImageData(im,0,0);return t;}
async function save(){const f=frames[fi];
  const blob=await new Promise(r=>classMaskPNG().toBlob(r,'image/png'));
  const res=await (await fetch('/api/save/'+f.id,{method:'POST',body:blob})).json();
  f.reviewed=true; document.getElementById('prog').textContent='сохранено: '+ru(res.corrected_class);
  next();}
function next(){if(fi<frames.length-1){fi++;loadFrame();}}
function prev(){if(fi>0){fi--;loadFrame();}}
document.onkeydown=e=>{if(e.target.tagName=='INPUT')return;
  if(e.key=='r')setTool('ord');else if(e.key=='f')setTool('fine');else if(e.key=='e')setTool('erase');
  else if(e.key=='1')recolorAll(1);else if(e.key=='2')recolorAll(2);else if(e.key=='0')eraseAll();
  else if(e.key==' '){e.preventDefault();toggleOverlay();}
  else if(e.key=='s'){e.preventDefault();save();}
  else if(e.key=='ArrowRight')next();else if(e.key=='ArrowLeft')prev();};
loadList();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed-dir", type=Path, default=None,
                    help="Dir of <frame_id>.png seed masks (e.g. model-A seeds) used as the initial state.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    store = Store(args.runs_dir, args.out_dir, args.seed_dir)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Grade mask review: http://{args.host}:{args.port}/")
    print(f"frames: {len(store.frames)}  reviewed: {sum(1 for f in store.frames if store.corrected_path(f['id']).exists())}")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
