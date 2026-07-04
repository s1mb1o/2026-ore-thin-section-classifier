#!/usr/bin/env python3
"""Render a Markdown deck + a separate CSS theme into ONE self-contained HTML file.

Content lives in a Markdown file (slides separated by lines containing only `---`);
the look lives in theme/deck.css (a separate file). Local images referenced with
![alt](path) or <img src="path"> are downscaled/recompressed and embedded as data
URIs, so the output HTML is a single portable, offline file (opens anywhere, prints
to PDF).

Usage:
  python render_presentation.py \
      --content presentation_ru.md --css theme/deck.css \
      --out presentation.html --title "..." --mode deck
  python render_presentation.py --content features_ru.md --css theme/deck.css \
      --out features.html --title "..." --mode page

Requires: markdown, Pillow (installed in the repo .venv).
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import re
from pathlib import Path

import markdown as md_lib
from PIL import Image

MAX_IMG_W = 1600  # embedded images are downscaled to at most this width


def _encode_image(path: Path) -> str | None:
    try:
        im = Image.open(path)
    except Exception as e:
        print(f"  ! could not open image {path}: {e}")
        return None
    im.load()
    if im.width > MAX_IMG_W:
        h = round(im.height * MAX_IMG_W / im.width)
        im = im.resize((MAX_IMG_W, h), Image.LANCZOS)
    name = path.name.lower()
    keep_png = im.mode in ("P", "1") or "mask" in name or "confidence" in name or (im.mode == "RGBA")
    buf = io.BytesIO()
    if keep_png:
        if im.mode not in ("RGB", "RGBA", "L", "P"):
            im = im.convert("RGB")
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def embed_images(html_text: str, base_dir: Path) -> str:
    """Replace <img src="local"> with data URIs (downscaled)."""
    def repl(m: re.Match) -> str:
        pre, src, post = m.group(1), m.group(2), m.group(3)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = (base_dir / src).resolve()
        data = _encode_image(p) if p.exists() else None
        if data is None:
            print(f"  ! missing image (kept as-is): {src}")
            return m.group(0)
        return f"<img{pre}src=\"{data}\"{post}>"

    return re.sub(r'<img([^>]*?)\ssrc="([^"]+)"([^>]*?)>', repl, html_text)


def mark_speaker_notes(html_text: str) -> str:
    """A paragraph that is entirely italic becomes a speaker note (Lenta convention)."""
    return re.sub(
        r'<p><em>(.*?)</em></p>',
        r'<p class="note"><em>\1</em></p>',
        html_text,
        flags=re.DOTALL,
    )


def color_arrows(html_text: str) -> str:
    """Colorize `->` inside <pre> code blocks so pipeline listings read as a flow."""
    def repl(m: re.Match) -> str:
        block = m.group(0)
        return block.replace("-&gt;", '<span class="arrow">→</span>')
    return re.sub(r'<pre>.*?</pre>', repl, html_text, flags=re.DOTALL)


def wrap_tables(html_text: str) -> str:
    return re.sub(r'(<table>.*?</table>)', r'<div class="table-wrap">\1</div>',
                  html_text, flags=re.DOTALL)


def make_md() -> md_lib.Markdown:
    return md_lib.Markdown(extensions=[
        "extra",        # tables, fenced_code, attr_list, md_in_html, etc.
        "sane_lists",
        "admonition",
    ])


def render_block(block: str, md: md_lib.Markdown, base_dir: Path) -> str:
    md.reset()
    out = md.convert(block)
    out = mark_speaker_notes(out)
    out = wrap_tables(out)
    out = color_arrows(out)
    out = embed_images(out, base_dir)
    return out


def is_appendix(block: str) -> bool:
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return "приложение" in s.lower() or "appendix" in s.lower()
    return False


PAGE_TMPL = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
{css}
</style></head>
<body class="{body_class}">
{body}
{script}
</body></html>
"""

DECK_SCRIPT = """<script>
(function(){
  var deck=document.querySelector('main.deck');
  var bar=document.querySelector('.progress');
  var counter=document.querySelector('.counter');
  var slides=[].slice.call(document.querySelectorAll('.slide'));
  function upd(){
    var st=window.scrollY, h=document.body.scrollHeight-window.innerHeight;
    if(bar) bar.style.width=(h>0?(st/h*100):0)+'%';
    var idx=0;
    for(var i=0;i<slides.length;i++){ if(slides[i].getBoundingClientRect().top<=window.innerHeight*0.5) idx=i; }
    if(counter) counter.textContent=(idx+1)+' / '+slides.length;
  }
  window.addEventListener('scroll',upd,{passive:true});
  window.addEventListener('resize',upd); upd();
  document.addEventListener('keydown',function(e){
    if(e.key==='n'||e.key==='N'){ document.body.classList.toggle('notes-off'); }
    if(e.key==='ArrowDown'||e.key==='PageDown'||e.key===' '){
      e.preventDefault(); var y=window.scrollY; for(var i=0;i<slides.length;i++){var t=slides[i].getBoundingClientRect().top; if(t>5){slides[i].scrollIntoView({behavior:'smooth'});return;}} }
    if(e.key==='ArrowUp'||e.key==='PageUp'){
      e.preventDefault(); for(var i=slides.length-1;i>=0;i--){var t=slides[i].getBoundingClientRect().top; if(t<-5){slides[i].scrollIntoView({behavior:'smooth'});return;}} }
  });
})();
</script>"""


def build_deck(blocks, md, base_dir, title):
    sections = []
    for i, blk in enumerate(blocks):
        cls = "slide"
        if i == 0:
            cls += " title"
        if is_appendix(blk):
            cls += " appendix"
        inner = render_block(blk, md, base_dir)
        sections.append(f'<section class="{cls}"><div class="slide-inner">{inner}</div></section>')
    header = (
        '<header class="bar">'
        '<span class="brand"><b>NORNICKEL</b> · Скажи мне, кто твой шлиф</span>'
        '<span class="hint">← ↑ ↓ → навигация · N — заметки спикера</span>'
        '<span class="counter"></span></header>'
        '<div class="progress"></div>'
    )
    body = header + '<main class="deck">' + "\n".join(sections) + "</main>"
    return body, DECK_SCRIPT, "deck"


def build_page(blocks, md, base_dir, title):
    head = render_block(blocks[0], md, base_dir) if blocks else ""
    cards = []
    for blk in blocks[1:]:
        span = " span2" if blk.strip().startswith("<!--span2-->") else ""
        cards.append(f'<article class="card{span}">{render_block(blk, md, base_dir)}</article>')
    body = (
        '<div class="page-wrap">'
        f'<div class="page-head">{head}</div>'
        f'<div class="features-grid">{"".join(cards)}</div>'
        '</div>'
    )
    return body, "", "page"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--css", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Презентация")
    ap.add_argument("--mode", choices=["deck", "page"], default="deck")
    args = ap.parse_args()

    content_path = Path(args.content).resolve()
    base_dir = content_path.parent
    raw = content_path.read_text(encoding="utf-8")
    css = Path(args.css).read_text(encoding="utf-8")

    blocks = [b.strip() for b in re.split(r'(?m)^---\s*$', raw) if b.strip()]
    md = make_md()

    if args.mode == "deck":
        body, script, body_class = build_deck(blocks, md, base_dir, args.title)
    else:
        body, script, body_class = build_page(blocks, md, base_dir, args.title)

    out_html = PAGE_TMPL.format(
        title=html.escape(args.title), css=css, body=body, script=script, body_class=body_class,
    )
    out_path = Path(args.out).resolve()
    out_path.write_text(out_html, encoding="utf-8")
    size_mb = len(out_html.encode("utf-8")) / 1e6
    print(f"wrote {out_path}  ({len(blocks)} blocks, {size_mb:.2f} MB, mode={args.mode})")


if __name__ == "__main__":
    main()
