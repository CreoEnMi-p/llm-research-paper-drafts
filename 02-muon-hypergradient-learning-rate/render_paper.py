"""Render PAPER.md to PAPER.html in the paper layout (single column, serif, centered figures).
Run: python3 render_paper.py"""
import re
import markdown

DISCLAIMER = ("<b>AI conference paper setup by Vuk Rosić.</b> The idea is judged to be novel, "
              "and with further work it may be submitted to a top AI conference.")
ONE_SENTENCE = ("<b>In one sentence:</b> multiplicative hypergradient descent, an old rule that grows the learning rate "
                "while each new gradient agrees with the last step and shrinks it when they disagree, lets the Muon "
                "optimizer find its learning rate in one training run, instead of training many copies at different "
                "rates, once it adapts only early and then holds the rate, makes each adjustment larger in wider "
                "models, and keeps weight decay on.")

STYLE = """<style>
html{background:#fbfbf9}
body{max-width:740px;margin:48px auto 80px;padding:0 20px;font-family:Georgia,"Times New Roman",serif;font-size:17px;line-height:1.55;color:#1d1d1d;background:#fbfbf9}
h1{text-align:center;font-size:1.9em;line-height:1.25;margin:0 0 .3em}
.byline{text-align:center;color:#444;margin:0 0 1.4em}
.topbox{border:1px solid #cfcfca;background:#f3f3ef;padding:12px 16px;margin:0 0 28px;font-size:.92em}
.topbox p{margin:.3em 0}
h2{font-size:1.3em;margin:1.8em 0 .5em}
h3{font-size:1.08em;margin:1.4em 0 .4em}
p{margin:.7em 0}
figure{margin:1.6em 0;text-align:center}
figure img{max-width:100%;height:auto}
figcaption{font-size:.9em;color:#333;text-align:left;margin:.5em auto 0;max-width:680px}
table{border-collapse:collapse;margin:1.2em auto;font-size:.82em;display:block;overflow-x:auto;width:fit-content;max-width:100%}
th,td{border:1px solid #c9c9c4;padding:4px 7px}
th{background:#efefea}
pre{background:#f1f1ec;padding:10px 12px;overflow-x:auto;font-size:.85em;line-height:1.4}
code{font-size:.9em}
</style>"""


def figures(html):
    # <p><img ...></p> followed by <p><em>Figure ...</em></p>  ->  <figure><img><figcaption>
    pat = re.compile(r'<p>(<img [^>]+>)</p>\s*<p><em>(Figure .*?)</em></p>', re.S)
    return pat.sub(r'<figure>\1<figcaption>\2</figcaption></figure>', html)


src = open("PAPER.md").read()
first, rest = src.split("\n", 1)
title = first.lstrip("# ").strip()
body = figures(markdown.markdown(rest, extensions=["tables", "fenced_code"]))
open("PAPER.html", "w").write(
    f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
    f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>{STYLE}</head><body>'
    f'<h1>{title}</h1><p class="byline">Vuk Rosić · AI Research School<br>November 2026</p>'
    f'<div class="topbox"><p>{DISCLAIMER}</p><p>{ONE_SENTENCE}</p></div>{body}</body></html>')
print("wrote PAPER.html")
