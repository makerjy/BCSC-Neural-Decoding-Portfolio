
"""
Geometric QA for the deck, standing in for a visual pass.

There is no LibreOffice in this environment, so the slides cannot be rendered and looked at.
What can still be checked mechanically is most of what a visual pass catches: shapes that
leave the margins, shapes that overlap each other, and text that is very likely to overflow
the box it was given. The last one is an estimate -- PowerPoint does the real line breaking --
so it is reported as a warning with the estimated overshoot rather than as a failure.

    python slides/qa_deck.py slides/neural_speech_decoding.pptx
"""
import sys
import unicodedata

from pptx import Presentation
from pptx.util import Emu

MARGIN = 0.5          # inches the design promises to keep clear
TOL = 0.02            # inches of slop before an overlap is called


def inches(v):
    return Emu(v).inches


def wide(ch):
    """Hangul and CJK occupy a full em; Latin about half."""
    return unicodedata.east_asian_width(ch) in ("W", "F")


def est_height(shape):
    """Rough rendered height of a text frame, in inches."""
    total = 0.0
    w = inches(shape.width)
    for p in shape.text_frame.paragraphs:
        runs = p.runs
        if not runs:
            continue
        size = max((r.font.size.pt if r.font.size else 18) for r in runs)
        s = "".join(r.text for r in runs)
        em = size / 72.0
        used = sum(em if wide(c) else em * 0.5 for c in s)
        lines = max(1, int(used / max(w, 1e-6) + 0.999))
        spacing = p.line_spacing if isinstance(p.line_spacing, float) else 1.2
        after = p.space_after.pt if p.space_after is not None else 0
        total += lines * em * spacing + after / 72.0
    return total


def rects(slide):
    out = []
    for sh in slide.shapes:
        r = (inches(sh.left), inches(sh.top),
             inches(sh.left) + inches(sh.width), inches(sh.top) + inches(sh.height))
        if sh.shape_type is not None and "AUTO_SHAPE" in str(sh.shape_type):
            kind, label = "panel", "card"          # tinted background, meant to sit under text
        elif sh.has_text_frame and sh.text_frame.text.strip():
            kind = "text"
            label = sh.text_frame.text.strip().split("\n")[0][:34]
        else:
            kind, label = "pic", "picture"
        out.append((r, kind, label, sh))
    return out


def main(path):
    prs = Presentation(path)
    W, H = inches(prs.slide_width), inches(prs.slide_height)
    fails = warns = 0

    for i, slide in enumerate(prs.slides, 1):
        items = rects(slide)
        for (x0, y0, x1, y1), kind, label, sh in items:
            if x0 < MARGIN - TOL or y0 < 0.35 - TOL or x1 > W - MARGIN + TOL \
                    or y1 > H - 0.35 + TOL:
                print("  FAIL slide %2d  outside margins  [%.2f %.2f %.2f %.2f]  %s"
                      % (i, x0, y0, x1, y1, label))
                fails += 1
            if kind == "text":
                need = est_height(sh)
                room = H - 0.35 - y0            # text grows downward past its box
                if need > room + TOL:
                    print("  FAIL slide %2d  text runs off the slide by %.2f in  %s"
                          % (i, need - room, label))
                    fails += 1
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (ax0, ay0, ax1, ay1), ka, la, _ = items[a]
                (bx0, by0, bx1, by1), kb, lb, _ = items[b]
                if "panel" in (ka, kb):
                    continue
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if ox > TOL and oy > TOL:
                    area = ox * oy
                    print("  WARN slide %2d  overlap %.2f x %.2f in  [%s: %s] / [%s: %s]"
                          % (i, ox, oy, ka, la, kb, lb))
                    warns += 1

    print("\n%d slides   %d failures   %d overlap warnings" % (len(prs.slides._sldIdLst),
                                                               fails, warns))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "slides/neural_speech_decoding.pptx"))
