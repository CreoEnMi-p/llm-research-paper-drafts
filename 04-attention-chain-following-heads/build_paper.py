"""PAPER.md -> PAPER.html (same style as the other near-finished papers)."""
import markdown, os
HERE = os.path.dirname(os.path.abspath(__file__))
md = open(os.path.join(HERE, "PAPER.md")).read()
title = md.splitlines()[0].lstrip("# ").strip()
body = markdown.markdown("\n".join(md.splitlines()[1:]), extensions=["tables"])
body = body.replace("<p>Vuk Rosić", "<p class=\"au\">Vuk Rosić", 1).replace("<p><em>Figure", "<p class=\"cap\"><em>Figure")
css = """body{font-family:Georgia,serif;max-width:740px;margin:40px auto;padding:0 16px;line-height:1.55;color:#222;background:#fff}
h1{font-size:27px;line-height:1.25;text-align:center}.au{text-align:center;color:#444}.cap{text-align:center;font-size:14px;color:#444}h2{margin-top:36px;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;margin:14px 0;font-size:14px;font-family:Helvetica,Arial,sans-serif}
th,td{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top}th{background:#f3f4f6}
img{max-width:100%;display:block;margin:12px auto}em{color:#444}code{background:#f3f4f6;padding:1px 4px;border-radius:3px;font-size:13px}"""
note = ('<p style="border:1px solid #c9c9c9;background:#f6f6f6;padding:10px 14px;margin:14px 0 22px;font-size:15px"><b>AI conference paper setup by Vuk Rosić.</b> '
        'The idea is judged to be novel, and with further work it may be submitted to a top AI conference.<br><br>'
        '<b>In one sentence:</b> a pointer head, an attention head that hands the word it found to the next layer, lets a transformer follow a chain of facts in its prompt such as cat→dog→sun→red with one layer per link, even when training checks only the final answer.'
        '</p>')
body = note + "\n" + body
html = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Following Chains of Facts</title>'
        f'<style>{css}</style></head><body><h1>{title}</h1>\n{body}\n</body></html>')
open(os.path.join(HERE, "PAPER.html"), "w").write(html); print("PAPER.html", len(html))
