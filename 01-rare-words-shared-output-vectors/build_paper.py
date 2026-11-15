"""Render PAPER.md to PAPER.html in the paper layout (skill section 0, "Page layout of PAPER.html")."""
import re
from pathlib import Path

import markdown

HERE = Path(__file__).parent
ONE_SENTENCE = ("In one sentence: in a language model's last layer each word has a vector that the model compares "
                "against to pick the next word; we let similar words share part of that vector, with groups the model "
                "finds itself from those vectors and redraws during training, and this helps rare words more than "
                "fixed groups or a tuned optimizer setting.")
DISCLAIMER = ("<b>AI conference paper setup by Vuk Rosić.</b> The idea is judged to be novel, and with further work "
              "it may be submitted to a top AI conference.")
CSS = """
:root{--ink:#1d1d1d;--muted:#555;--rule:#d4d4d4;--box:#f5f5f2;--bg:#fdfdfb}
body{max-width:740px;margin:48px auto;padding:0 16px;font-family:Georgia,"Times New Roman",serif;
 font-size:17px;line-height:1.55;color:var(--ink);background:var(--bg)}
h1{text-align:center;font-size:1.75em;line-height:1.25;margin:0 0 .4em}
.meta{text-align:center;color:var(--muted);font-style:italic;margin:0 0 1.2em}
.top{border:1px solid var(--rule);background:var(--box);padding:12px 16px;margin:0 0 1.8em;font-size:.95em}
.top p{margin:.3em 0}
h2{font-size:1.25em;margin:1.8em 0 .5em}
h2.abs{text-align:center;font-size:1.1em}
p{text-align:justify;hyphens:auto}
figure{margin:1.6em auto;text-align:center}
figure img{max-width:100%;height:auto}
figcaption{font-size:.9em;color:var(--muted);margin-top:.4em;text-align:center}
table{border-collapse:collapse;margin:1em auto;font-size:.85em;display:block;overflow-x:auto;width:fit-content;max-width:100%}
th,td{border:1px solid var(--rule);padding:4px 7px;text-align:left}
th{background:var(--box)}
pre{background:var(--box);padding:10px;overflow-x:auto;font-size:.82em}
code{font-size:.9em}
"""

md = (HERE / "PAPER.md").read_text()
lines = md.splitlines()
title = lines[0].lstrip("# ").strip()
body = markdown.markdown("\n".join(lines[1:]), extensions=["tables", "fenced_code"])
# image paragraph followed by an italic caption paragraph -> centered figure with caption below
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" /></p>\s*<p><em>(.*?)</em></p>',
              r'<figure><img alt="\1" src="\2"><figcaption>\3</figcaption></figure>', body, flags=re.S)
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" /></p>',
              r'<figure><img alt="\1" src="\2"></figure>', body)
body = body.replace("<h2>Abstract</h2>", '<h2 class="abs">Abstract</h2>', 1)
html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>'
        f'<style>{CSS}</style></head><body><h1>{title}</h1>'
        f'<p class="meta">Vuk Rosić · AI Research School · November 2026</p>'
        f'<div class="top"><p>{DISCLAIMER}</p><p>{ONE_SENTENCE}</p></div>{body}</body></html>')
(HERE / "PAPER.html").write_text(html)
print("PAPER.html written;", html.count("<figure>"), "figures")
