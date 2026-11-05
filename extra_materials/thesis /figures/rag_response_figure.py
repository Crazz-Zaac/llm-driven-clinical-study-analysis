"""Render an excerpt of an *all papers* response as a readable UI-style figure.

The answers are the stored Llama 3.2 (3B) responses of evaluation run V1, so
the figure shows real system output. The table is laid out as in the chat UI
(Paper | Answer | Evidence; answers shown as returned, including raw markdown)
but at a width that stays legible when scaled to \\linewidth.

Usage:  python rag_response_figure.py
Needs:  firefox (headless screenshot), pillow
Writes: RAG_response.png next to this script.
"""

import ast
import csv
import glob
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "llm4ehr/rag_evaluation/manually_reviewed/results_V1"
QUESTION_PREFIX = "What patient population was included"
PAPER_INDICES = [9, 6, 1, 7]  # plain sentence, bullet list, nested table, not reported
WIDTH_CSS = 900  # page width in CSS px; ~7 pt answer text at \linewidth
SCALE = 2  # CSS zoom factor of the screenshot
OUT = HERE / "RAG_response.png"


def load_rows():
    rows = []
    for f in glob.glob(str(RESULTS / "llama_3_2b_V1_*.csv")):
        rows += list(csv.DictReader(open(f, encoding="utf-8")))
    sel = {int(r["paper_index"]): r for r in rows
           if r["user_input"].startswith(QUESTION_PREFIX)}
    assert len(sel) == 12, f"expected 12 papers, found {len(sel)}"
    return sel


def paper_cells(r):
    ctx = ast.literal_eval(r["retrieved_contexts"])[0]
    title = re.search(r"^TITLE: (.*)$", ctx, re.M).group(1)
    sections = re.search(r"^SECTIONS: (.*)$", ctx, re.M).group(1)
    # undo the table-cell escaping applied by the pipeline (pipeline.py)
    answer = r["response"].replace("\\|", "|").replace("<br>", "\n").strip()
    return title, answer, sections


def build_html(rows):
    first = rows[PAPER_INDICES[0]]
    question = first["user_input"].split("\n\nEvaluate only this paper")[0]
    model = first["response_model"]
    body = "\n".join(
        f"<tr><td class='paper'>{html.escape(t)}</td>"
        f"<td class='answer'>{html.escape(a)}</td>"
        f"<td class='evidence'>{html.escape(s)}</td></tr>"
        for t, a, s in (paper_cells(rows[i]) for i in PAPER_INDICES))
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  html {{ zoom: {SCALE}; }}
  html, body {{ margin: 0; background: #ffffff; }}
  body {{ width: {WIDTH_CSS}px; padding: 16px; box-sizing: border-box;
         font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
         font-size: 15px; color: #1f2933; line-height: 1.45; }}
  .bar {{ display: flex; gap: 10px; align-items: center; margin-bottom: 14px;
          font-size: 13px; color: #52606d; }}
  .chip {{ border: 1px solid #cbd2d9; border-radius: 6px; padding: 3px 9px;
           background: #f5f7fa; color: #1f2933; }}
  .q {{ margin-left: auto; max-width: 78%; background: #0b7285; color: #fff;
        border-radius: 14px; padding: 10px 14px; margin-bottom: 14px; }}
  .card {{ border: 1px solid #d9e2ec; border-radius: 10px; overflow: hidden; }}
  table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  th {{ text-align: left; background: #f0f4f8; font-weight: 600;
        padding: 8px 10px; border-bottom: 1px solid #d9e2ec; }}
  td {{ vertical-align: top; padding: 9px 10px; border-bottom: 1px solid #e4e7eb; }}
  tr:last-child td {{ border-bottom: none; }}
  td.paper {{ color: #1c6dd0; }}
  td.answer {{ white-space: pre-wrap; font-size: 14px; }}
  td.evidence {{ color: #52606d; }}
  .note {{ margin-top: 8px; font-size: 12px; color: #7b8794; }}
</style></head><body>
  <div class="bar"><span>Search scope</span><span class="chip">All indexed papers</span>
    <span>Chat model</span><span class="chip">{html.escape(model)} (local)</span></div>
  <div class="q">{html.escape(question)}</div>
  <div class="card"><table>
    <colgroup><col style="width:27%"><col style="width:58%"><col style="width:15%"></colgroup>
    <tr><th>Paper</th><th>Answer</th><th>Evidence</th></tr>
    {body}
  </table></div>
  <div class="note">{len(PAPER_INDICES)} of {len(rows)} rows shown</div>
</body></html>"""


def screenshot(page: Path, out: Path):
    firefox = shutil.which("firefox")
    assert firefox, "firefox not found on PATH"
    # the page is scaled with CSS zoom, so the window is SCALE times wider
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([firefox, "--headless", "--no-remote", "--profile", prof,
                        "--screenshot", str(out),
                        f"--window-size={WIDTH_CSS * SCALE},{3000 * SCALE}",
                        page.as_uri()], check=True, capture_output=True, timeout=120)
    # trim the empty white area below the content
    img = Image.open(out).convert("RGB")
    bbox = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    img.crop((0, 0, img.width, min(img.height, bbox[3] + 16 * SCALE))).save(out)


if __name__ == "__main__":
    rows = load_rows()
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp, "rag_response.html")
        page.write_text(build_html(rows), encoding="utf-8")
        screenshot(page, OUT)
    w, h = Image.open(OUT).size
    print(f"wrote {OUT.name} ({w}x{h})")
