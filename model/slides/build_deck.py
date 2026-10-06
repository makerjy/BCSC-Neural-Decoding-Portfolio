
"""
Builds the presentation from slides/figs/.

Styled after latex.vercel.app: white page, serif throughout, no decorative rules or colour
bars, generous margins, one accent colour used only for "ours". The figures are already in
Computer Modern; the slide text uses Times New Roman for Latin and Batang for Hangul, which is
the closest pairing that ships on the machines this will actually be shown from.

    python slides/build_deck.py
"""
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
OUTFILE = os.path.join(HERE, "neural_speech_decoding.pptx")

W, H = 13.333, 7.5
MARGIN = 0.9

INK = RGBColor(0x11, 0x11, 0x11)
MUTED = RGBColor(0x77, 0x77, 0x77)
FAINT = RGBColor(0xAA, 0xAA, 0xAA)
ACC = RGBColor(0xA6, 0x2B, 0x1F)
ACC2 = RGBColor(0x20, 0x54, 0x8F)

SERIF = "Times New Roman"
HANGUL = "바탕"


def _font(run, size, color=INK, bold=False, italic=False, face=SERIF):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = face
    rPr = run.font._rPr                       # Hangul needs its own typeface entry
    for tag in ("a:ea", "a:cs"):
        e = rPr.makeelement(qn(tag), {"typeface": HANGUL if tag == "a:ea" else face})
        rPr.append(e)


def text(slide, x, y, w, h, blocks, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         line=1.25, space=6):
    """blocks: list of (string, size, colour, bold, italic) or (string, size) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, b in enumerate(blocks):
        s, size = b[0], b[1]
        color = b[2] if len(b) > 2 else INK
        bold = b[3] if len(b) > 3 else False
        ital = b[4] if len(b) > 4 else False
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt(space)
        _font(p.add_run(), size, color, bold, ital)
        p.runs[0].text = s
    return tb


def picture(slide, name, box):
    """Fit figs/<name>.png inside box=(x, y, w, h) preserving aspect, centred."""
    path = os.path.join(FIGS, name + ".png")
    iw, ih = Image.open(path).size
    x, y, bw, bh = box
    scale = min(bw / iw, bh / ih)
    w, h = iw * scale, ih * scale
    slide.shapes.add_picture(path, Inches(x + (bw - w) / 2), Inches(y + (bh - h) / 2),
                             width=Inches(w), height=Inches(h))


def card(slide, x, y, w, h, tint):
    """A flat tinted panel. Tint only -- no edge stripe, no border rule."""
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = tint
    sh.line.fill.background()
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def head(slide, title, kicker=None):
    y = 0.52
    if kicker:
        text(slide, MARGIN, y, W - 2 * MARGIN, 0.3, [(kicker, 13, MUTED)])
        y += 0.38
    text(slide, MARGIN, y, W - 2 * MARGIN, 0.5, [(title, 27)])
    return y + 0.95


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


# ---------------------------------------------------------------- slides

def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # 1 ---------------------------------------------------------------- title
    s = blank(prs)
    text(s, MARGIN, 2.35, W - 2 * MARGIN, 1.0,
         [("뇌파에서 음소를 읽어내기", 40)], align=PP_ALIGN.CENTER)
    text(s, MARGIN, 3.35, W - 2 * MARGIN, 0.6,
         [("설계한 요소가 실제로 그 이유로 작동하는가", 17, MUTED, False, True)],
         align=PP_ALIGN.CENTER)
    text(s, MARGIN, 4.55, W - 2 * MARGIN, 1.2,
         [("11개 음소 분류  ·  16명  ·  61채널 EEG", 14),
          ("OpenNeuro ds006104", 12, MUTED)],
         align=PP_ALIGN.CENTER, space=8)
    notes(s, "발표 시작. 우리가 만든 것은 정확도 경쟁 모델이 아니라, "
             "'왜 이 부품이 있는가'에 답할 수 있는 모델이라는 점을 먼저 말한다.")

    # 2 ---------------------------------------------------------------- headline
    s = blank(prs)
    head(s, "한 장 요약")
    cols = [("0.306", "11음소 정확도", "우연 0.091의 3.4배", ACC),
            ("15 / 15", "베이스라인 대비", "동일 폴드 전승,  p = 2×10⁻⁸", INK),
            ("+0.000", "선택 편향", "미접촉 test셋에서 측정", INK)]
    for i, (big, lab, sub, c) in enumerate(cols):
        x = MARGIN + i * (W - 2 * MARGIN) / 3
        text(s, x, 2.45, (W - 2 * MARGIN) / 3 - 0.4, 1.6,
             [(big, 54, c), (lab, 15), (sub, 12, MUTED)], space=7)
    text(s, MARGIN, 4.95, W - 2 * MARGIN, 1.6,
         [("그런데 이 발표의 내용은 위 숫자가 아니다.", 17),
          ("우리가 중요하다고 생각한 요소 — 시간 구조 — 를 모델 안에 직접 넣었고, "
           "그것이 주장한 이유로 작동한다는 것을 같은 차원의 대조군으로 보였다.", 15, MUTED)],
         space=10)
    notes(s, "숫자는 빠르게 넘기고 '메커니즘' 문장에 시간을 쓴다. "
             "11클래스에서 raw 퍼센트는 오해를 만들므로 항상 우연 대비로 말할 것.")

    # 3 ---------------------------------------------------------------- data
    s = blank(prs)
    y = head(s, "데이터와 과제", "1. 배경")
    picture(s, "montage", (MARGIN, y - 0.1, 7.2, 4.3))
    rows = [("자극", "음소 하나를 듣는다  (a i e u o s z p b t d)"),
            ("피험자", "16명 × 약 208시행 = 3328"),
            ("채널", "61개, ICA 정제"),
            ("창", "0 ~ 1.0초, 256 Hz"),
            ("우연 수준", "1/11 = 0.091")]
    yy = y + 0.25
    for k, v in rows:
        text(s, 8.5, yy, 1.4, 0.32, [(k, 13, MUTED)])
        text(s, 9.9, yy, 2.6, 0.32, [(v, 13)])
        yy += 0.52
    text(s, 8.5, yy + 0.25, 4.0, 1.1,
         [("자극 전 구간이 없다.", 13, ACC),
          ("그래서 유발반응의 DC 성분이 살아남는 자리가 모델 안에 있어야 한다 — "
           "이것이 설계 전체를 규정한다.", 12, MUTED)], space=6)
    notes(s, "baseline 보정이 불가능하다는 점을 강조. 뒤에 나오는 '중심화를 하지 않는다'와 "
             "'시간 기저 행'이 전부 여기서 나온다.")

    # 4 ---------------------------------------------------------------- hard
    s = blank(prs)
    y = head(s, "왜 어려운가", "1. 배경")
    items = [("단일 시행 효과크기", "최대 |Cohen d| = 0.98", "신호는 실재하지만 잡음이 압도한다"),
             ("시행을 평균하면", "1회 0.550 → 8회 0.585", "8배를 모아도 조금 오른다"),
             ("11클래스 산술", "0.6이 되려면 이진 대비가 0.9", "우리 최고 이진이 0.650")]
    yy = y + 0.15
    for k, v, note in items:
        text(s, MARGIN, yy, 3.6, 0.4, [(k, 15)])
        text(s, 4.7, yy, 3.4, 0.4, [(v, 15, ACC)])
        text(s, 8.3, yy, 4.2, 0.4, [(note, 13, MUTED, False, True)])
        yy += 1.05
    text(s, MARGIN, yy + 0.35, W - 2 * MARGIN, 1.0,
         [("정확도를 크게 올리는 길은 더 좋은 모델이 아니라 더 많은 데이터다.", 17),
          ("그래서 우리는 정확도 대신 '무엇이 신호를 만드는가'를 물었다.", 14, MUTED)],
         space=8)
    notes(s, "여기서 발표의 방향 전환을 선언한다. 성능 경쟁이 아니라 메커니즘 규명이다.")

    # 5 ---------------------------------------------------------------- key idea
    s = blank(prs)
    y = head(s, "핵심 문제 — 시간을 요약하면 진다", "2. 설계")
    text(s, MARGIN, y, 5.6, 3.0,
         [("시행 하나는 61채널 × 256샘플의 시계열이다 (필터를 지나면 14 × 256). "
           "고정 길이 특징으로 바꿔야 하는데, 보통 쓰는 공분산 C = ZZᵀ/T 는 "
           "시간에 대해 평균을 내버린다.", 15),
          ("\"어느 채널이 함께 움직이나\"는 남고 \"언제\"는 사라진다.", 14, MUTED, False, True)],
         space=12)
    losers = [("대역파워 (Welch)", "0.106"), ("클래스 개념자", "0.099"),
              ("필터뱅크 Riemannian", "0.097"), ("시간-주파수 (STFT)", "0.123"),
              ("전체 궤적 공분산", "시간 평균")]
    text(s, 7.3, y - 0.05, 5.0, 0.4, [("시간을 요약으로 소거한 표상들 (11음소)", 13, MUTED)])
    yy = y + 0.45
    for k, v in losers:
        text(s, 7.3, yy, 3.4, 0.32, [(k, 13)])
        text(s, 10.9, yy, 1.4, 0.32, [(v, 13, MUTED)], align=PP_ALIGN.RIGHT)
        yy += 0.44
    text(s, MARGIN, y + 3.3, 11.5, 0.9,
         [("우리의 답 — 행을 더 쌓고 나서 공분산을 만들면, 공분산 하나 안에 시간 구조가 들어온다.",
           17, ACC)])
    notes(s, "이 슬라이드가 발표의 전환점이다. 문제 정의를 명확히 하고 다음 슬라이드에서 해법을 보인다.")

    # 6 ---------------------------------------------------------------- architecture
    s = blank(prs)
    head(s, "설계본의 구조", "2. 설계")
    picture(s, "arch", (MARGIN, 1.50, W - 2 * MARGIN, 4.35))
    text(s, MARGIN, 6.15, W - 2 * MARGIN, 0.7,
         [("STAGE 2가 이 설계의 핵심이며, 세 단 중 주장한 메커니즘이 통제를 통과한 유일한 부품이다.",
           14, MUTED)])
    notes(s, "STAGE 1은 신호대잡음 최대화(분모가 잡음). STAGE 2가 본체. "
             "STAGE 3은 피험자를 분리하는 게 아니라 섞지 않는 것.")

    # 7 ---------------------------------------------------------------- object
    s = blank(prs)
    y = head(s, "객체 안에는 무엇이 들어 있나", "2. 설계")
    picture(s, "spd", (MARGIN - 0.3, y - 0.2, 5.6, 4.9))
    text(s, 6.6, y, 5.9, 3.6,
         [("Z × Z", 16, INK, True),
          ("어느 성분들이 함께 움직이나 — 공간 패턴", 13, MUTED),
          ("Z × Z(t−τ)", 16, INK, True),
          ("지금과 78 ms 전의 관계 — 동역학", 13, MUTED),
          ("Z × φ", 16, ACC, True),
          ("각 성분의 파형이 k번째 시계와 얼마나 닮았나 — 시간분해된 유발반응", 13, MUTED)],
         space=4)
    text(s, 6.6, y + 3.75, 5.9, 0.9,
         [("φ 행은 모든 시행에 대해 똑같은 고정 파형이다. 이것이 시간 순서를 "
           "객체 안으로 들여보내는 통로다.", 13)])
    notes(s, "실제 데이터로 계산한 행렬이다. 블록 경계선이 14, 28에 있다.")

    # 8 ---------------------------------------------------------------- phi rows
    s = blank(prs)
    y = head(s, "시계(clock) — φ 행", "2. 설계")
    picture(s, "phi", (MARGIN, y - 0.1, 7.0, 4.4))
    text(s, 8.3, y + 0.3, 4.2, 3.4,
         [("φ(k, t) = cos(π k t),   k = 0 … 7", 15),
          ("직교 DCT 기저, unit RMS", 13, MUTED),
          ("Z 를 φ 와 곱해 시간 평균하면 그 성분 파형의 k번째 주파수 성분이 나온다.", 13),
          ("K = 7 이면 1.01초 창에서 최고 성분이 약 3.5 Hz — N1/P2 유발복합체의 시간 척도.",
           13, ACC)], space=10)
    notes(s, "φ가 '추가 특징'이 아니라 SPD 객체의 기하를 바꾸는 물건이라는 점이 중요하다.")

    # 9 ---------------------------------------------------------------- control 1
    s = blank(prs)
    head(s, "통제 — 시간 구조가 진짜로 일하는가", "3. 통제")
    picture(s, "phi_ablation", (MARGIN, 1.50, 7.6, 4.50))
    text(s, 8.9, 1.7, 3.6, 4.0,
         [("차원을 똑같이 맞춘 채 시간 구조만 파괴했다.", 14),
          ("정확도가 오른 것과 주장한 메커니즘이 작동한 것은 다르다.", 13, MUTED, False, True),
          ("차원 효과가 아니고,", 13),
          ("아무 행이나 되는 것도 아니고,", 13),
          ("시간 척도가 맞아야 한다.", 14, ACC)], space=10)
    notes(s, "막대 위의 점이 폴드별 값이다. 네 대조군 모두 같은 666차원.")

    # 10 --------------------------------------------------------------- control 2
    s = blank(prs)
    y = head(s, "통제 — 시간 척도가 맞아야 한다", "3. 통제")
    picture(s, "ktime", (MARGIN, y - 0.1, 6.0, 4.4))
    text(s, 7.6, y + 0.5, 4.9, 3.2,
         [("행을 늘리기만 하면 좋아지지 않는다.", 15),
          ("K를 0에서 23까지 늘리면 정확도가 역 U자를 그린다. 최적 K=7이 만드는 "
           "최고 성분 주파수가 유발반응의 시간 척도와 일치한다.", 13),
          ("모델이 붙잡은 것이 우리가 붙잡으려던 것과 같다는 뜻이다.", 14, ACC)], space=12)
    notes(s, "용량-반응 곡선은 '많을수록 좋다'가 아니라는 것을 보이는 표준적인 논증이다.")

    # 11 --------------------------------------------------------------- best model
    s = blank(prs)
    head(s, "최고 성능 모델", "4. 성적")
    picture(s, "bands", (MARGIN, 1.50, W - 2 * MARGIN, 4.40))
    text(s, MARGIN, 6.15, W - 2 * MARGIN, 0.7,
         [("객체 하나는 T = 236 표본에서 추정되므로 변이 약 40을 넘을 수 없다. "
           "그래서 크게 만들지 않고 밴드로 나눠 여러 개 만든다.", 14, MUTED)])
    notes(s, "이건 설계가 아니라 앙상블이라는 점을 반드시 밝힌다. 두 표현이 서로 다른 정보를 "
             "담고 있어서 이기는 것이지, 구조적 근거가 있는 것이 아니다.")

    # 12 --------------------------------------------------------------- results
    s = blank(prs)
    y = head(s, "성적", "4. 성적")
    picture(s, "results", (MARGIN - 0.2, y - 0.1, 6.0, 4.5))
    picture(s, "paired", (7.0, y - 0.1, 5.3, 4.5))
    text(s, MARGIN, y + 4.4, W - 2 * MARGIN, 0.7,
         [("왼쪽: 한 번도 만지지 않은 최종 test셋.   오른쪽: 베이스라인과 동일 폴드에서 짝지어 비교.",
           13, MUTED)])
    notes(s, "선택 편향 +0.000은 inner CV가 고를 모델과 실제 최고가 같았다는 뜻이다.")

    # 13 --------------------------------------------------------------- leakage
    s = blank(prs)
    y = head(s, "누수는 없는가", "5. 검증")
    picture(s, "splits", (MARGIN, y - 0.1, 7.4, 4.5))
    text(s, 8.8, y + 0.3, 3.7, 3.6,
         [("같은 녹음이 양쪽에 들어가지 않게 나눠도,", 13),
          ("시간 블록으로 나눠도 성적이 유지된다.", 13),
          ("라벨을 섞으면 정확히 우연으로 떨어진다.", 14, ACC),
          ("필터가 라벨을 쓰는 지도 학습인데도 그렇다 — 모든 적합이 훈련 폴드 안에서만 "
           "일어난다는 뜻이다.", 12, MUTED)], space=10)
    notes(s, "순열 귀무가 0.088로 우연 0.091에 앉는 것이 가장 강한 증거다.")

    # 14 --------------------------------------------------------------- interpretation
    s = blank(prs)
    y = head(s, "모델은 무엇을 보고 있나", "5. 검증")
    picture(s, "patterns", (MARGIN, y, 11.5, 2.0))
    picture(s, "confusion", (MARGIN + 0.4, y + 2.15, 4.6, 3.0))
    text(s, 6.6, y + 2.4, 5.9, 2.6,
         [("오분류가 음성학을 따른다.", 15),
          ("모음과 자음이 블록을 이루고, 유성/무성 짝끼리 혼동된다 (s↔z, t↔d, p↔b).", 13),
          ("자질 정보는 있으나 음소 정체를 완전히 가르기에는 부족하다.", 13, MUTED)],
         space=10)
    notes(s, "위: 공간 필터를 패턴으로 변환해 그린 것(Haufe et al. 2014). "
             "필터를 그대로 그리면 해석이 틀린다.")

    # 15 --------------------------------------------------------------- generalisation
    s = blank(prs)
    y = head(s, "미학습 피험자에게는 통하지 않는다", "6. 일반화")
    picture(s, "budget", (MARGIN, y - 0.1, 6.4, 4.4))
    text(s, 7.9, y + 0.3, 4.6, 3.6,
         [("0.306은 \"그 사람의 보정 데이터가 있을 때\"의 값이다.", 14),
          ("새로운 사람에게는 0.143 — 어떤 모델로도 동작하지 않고, 딥과 고전이 동률이다.",
           14, ACC),
          ("이유를 측정했다: 피험자들의 판별 방향이 사실상 직교한다 (코사인 0.01, "
           "같은 사람 재현은 0.45).", 13, MUTED)], space=10)
    notes(s, "두 숫자를 섞어 쓰지 말 것. 질문이 나오면 이 슬라이드로 돌아온다.")

    # 16 --------------------------------------------------------------- falsified
    s = blank(prs)
    y = head(s, "우리가 스스로 반증한 것", "7. 정직성")
    picture(s, "emergence", (MARGIN - 0.3, y, 6.2, 3.5))
    picture(s, "reservoir_sweep", (6.7, y, 5.6, 3.5))
    text(s, MARGIN, y + 3.6, 11.5, 1.4,
         [("설계를 대체하려는 두 방향을 모두 시험했다 — 역전파로 학습시키기, "
           "그리고 랜덤 동역학에서 창발시키기. 둘 다 졌다.", 15),
          ("리저버는 ρ가 커질수록, 크기를 키울수록 나빠진다. 최적점이 "
           "\"리저버가 없는 쪽\" 모서리에 앉아 있다.", 13, MUTED)], space=8)
    notes(s, "대부분의 포스터는 통과한 것만 보여준다. 반증에 숫자와 이유가 붙어 있다는 게 "
             "우리 작업의 값이다.")

    # 17 --------------------------------------------------------------- audit
    s = blank(prs)
    y = head(s, "통제 감사 — 항목별 집계", "7. 정직성")
    groups = [("통과", "4", "시간 기저 · 지도 공간 투영 · SNR 분모 · 개인 판독",
               RGBColor(0x2E, 0x6B, 0x4F), RGBColor(0xEE, 0xF4, 0xF0)),
              ("축소", "2", "라플라시안 기저 · 접선 기저",
               RGBColor(0x8A, 0x6D, 0x1F), RGBColor(0xF7, 0xF4, 0xEA)),
              ("반증", "6", "필터의 피험자/음소 분리 · 비선형 지연 동역학 · 리저버 계보 · "
                            "τ · 피험자 통합 · 기하 내 연결주의적 창발",
               ACC, RGBColor(0xFA, 0xF0, 0xEF))]
    yy = y + 0.15
    for name, n, items, c, tint in groups:
        card(s, MARGIN, yy, 11.53, 1.15, tint)
        text(s, MARGIN + 0.35, yy + 0.18, 1.2, 0.8, [(n, 40, c)])
        text(s, MARGIN + 1.55, yy + 0.36, 1.6, 0.45, [(name, 18, c)])
        text(s, MARGIN + 3.3, yy + 0.24, 7.9, 0.75, [(items, 13, MUTED)])
        yy += 1.30
    text(s, MARGIN, yy + 0.15, 11.5, 0.6,
         [("라벨 누수도 두 건 잡았다 — 클래스 짝맞춤 대리변수(+0.028 부풀림)와 "
           "과거 hyperalignment (0.991 → 0.531).", 14)])
    notes(s, "이 표가 발표의 정체성이다. 시간이 없으면 다른 슬라이드를 버리고 이걸 남긴다.")

    # 18 --------------------------------------------------------------- limits
    s = blank(prs)
    y = head(s, "한계", "8. 한계")
    lim = [("절대 성능", "11클래스 0.306은 실용 근처가 아니다"),
           ("미학습 피험자", "0.14 — 보정 프로토콜을 최적화해도 쓸 만해지지 않는다"),
           ("학습곡선", "아직 상승 중. 이 데이터셋은 사람당 208시행이 끝이다"),
           ("음향 대 음소", "수동 청취 설계에서는 원리적으로 분리할 수 없다")]
    yy = y + 0.15
    for k, v in lim:
        text(s, MARGIN, yy, 3.4, 0.5, [(k, 16)])
        text(s, 4.6, yy, 7.9, 0.6, [(v, 14, MUTED)])
        yy += 0.95
    text(s, MARGIN, yy + 0.25, 11.5, 0.8,
         [("정확도를 더 올리려면 데이터를 바꿔야 한다 — 사람당 시행 수를 늘리거나 "
           "상상·발화 과제로 이동.", 15, ACC)])
    notes(s, "한계를 먼저 말하면 질의응답이 훨씬 편해진다.")

    # 19 --------------------------------------------------------------- close
    s = blank(prs)
    text(s, MARGIN, 2.5, W - 2 * MARGIN, 1.2,
         [("정확도가 오른 것과", 26, MUTED),
          ("주장한 메커니즘이 작동한 것은 다르다.", 26)],
         align=PP_ALIGN.CENTER, space=10)
    text(s, MARGIN, 4.6, W - 2 * MARGIN, 1.2,
         [("우리는 부품마다 동일 차원·동일 규모의 대조군을 만들어 그 차이를 측정했다.",
           15, MUTED)], align=PP_ALIGN.CENTER)
    notes(s, "마무리 문장. 여기서 질문을 받는다.")

    prs.save(OUTFILE)
    print("wrote %s  (%d slides)" % (os.path.relpath(OUTFILE, os.path.dirname(HERE)),
                                     len(prs.slides._sldIdLst)))


if __name__ == "__main__":
    build()
