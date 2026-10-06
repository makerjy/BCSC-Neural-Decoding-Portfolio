
"""
Approximate renderer, so the deck can actually be looked at.

There is no LibreOffice here, so `soffice --convert-to pdf` is unavailable. This reads the
built .pptx back -- shape positions, text, sizes, colours, images -- and redraws it with
matplotlib. It is not a faithful renderer (PowerPoint's line breaking and font metrics differ,
and Hangul falls back to whatever matplotlib has), but it is read from the real file rather
than from the build script, so it catches the things a visual pass is for: a figure sitting on
top of a caption, a column that does not line up, a slide that is too full or too empty.

    python slides/preview.py slides/neural_speech_decoding.pptx
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pptx import Presentation
from pptx.util import Emu

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview")


def inches(v):
    return Emu(v).inches


def rgb(c):
    try:
        return "#%02X%02X%02X" % (c.rgb[0], c.rgb[1], c.rgb[2])
    except Exception:
        return "#111111"


def render(prs, idx, slide, W, H):
    fig = plt.figure(figsize=(W, H), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor="#DDDDDD"))

    for sh in slide.shapes:
        x, y = inches(sh.left), inches(sh.top)
        w, h = inches(sh.width), inches(sh.height)
        if sh.shape_type == 13 or sh.__class__.__name__ == "Picture":
            try:
                im = Image.open(io.BytesIO(sh.image.blob))
                ax.imshow(im, extent=(x, x + w, y + h, y), aspect="auto", zorder=2)
            except Exception:
                ax.add_patch(Rectangle((x, y), w, h, facecolor="#EEEEEE", edgecolor="#999999"))
            continue
        if sh.shape_type is not None and "AUTO_SHAPE" in str(sh.shape_type):
            try:
                fc = "#%02X%02X%02X" % tuple(sh.fill.fore_color.rgb)
            except Exception:
                fc = "#F2F2F2"
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="none", zorder=1))
            continue
        if not sh.has_text_frame:
            continue
        cy = y
        for p in sh.text_frame.paragraphs:
            runs = p.runs
            if not runs:
                continue
            size = max((r.font.size.pt if r.font.size else 18) for r in runs)
            txt = "".join(r.text for r in runs)
            col = rgb(runs[0].font.color)
            ital = "italic" if runs[0].font.italic else "normal"
            # crude wrap at the box width, full-width characters counted as one em
            per = max(1, int(w * 72 / (size * 0.62)))
            lines, cur = [], ""
            for ch in txt:
                cur += ch
                if len(cur) >= per:
                    lines.append(cur); cur = ""
            if cur:
                lines.append(cur)
            ha = {1: "left", 2: "center", 3: "right"}.get(
                int(p.alignment) if p.alignment is not None else 1, "left")
            ax_x = x if ha == "left" else (x + w / 2 if ha == "center" else x + w)
            for ln in lines:
                cy += size / 72.0 * 1.15
                ax.text(ax_x, cy, ln, fontsize=size * 0.82, color=col, ha=ha,
                        va="baseline", style=ital, zorder=3)
            cy += (p.space_after.pt if p.space_after is not None else 0) / 72.0

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "slide-%02d.png" % idx)
    fig.savefig(path, dpi=72)
    plt.close(fig)
    return path


def main(path):
    prs = Presentation(path)
    W, H = inches(prs.slide_width), inches(prs.slide_height)
    for i, slide in enumerate(prs.slides, 1):
        print(render(prs, i, slide, W, H))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "slides/neural_speech_decoding.pptx")
