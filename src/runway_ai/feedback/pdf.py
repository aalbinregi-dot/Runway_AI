"""Printable PDF assessment.

Design notes
------------
The page echoes the app: monochrome, high-contrast serif display type, and
score quality carried by **luminance** rather than colour. That is not only an
aesthetic choice - it means the report survives being printed in black and
white, which is how a coach will actually receive it.

reportlab ships Times and Helvetica; it does not ship Bodoni or Didot. Times-Roman
is the honest closest match for the display face rather than shipping a font
file, and Helvetica stands in for the interface grotesque.

The illustration is a **skeleton drawn from the stored keypoints**, not a video
still. Two reasons, in order of importance: no footage is kept, so a still is
not available for a walk pulled from history; and a stick figure carries the
measured geometry without carrying the subject's likeness into a document that
is meant to be shared.
"""
from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas

from ..config import CRITERIA, CRITERION_LABELS, CRITERION_WEIGHTS

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

INK = colors.HexColor("#161719")
MUTED = colors.HexColor("#6b7075")
FAINT = colors.HexColor("#b8bcc0")
RULE = colors.HexColor("#d8dade")

BONES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def _grey(score: float) -> colors.Color:
    """Luminance stands in for quality: darker is better.

    A 10/10 prints near-black and a 0/10 near-white, so the ranking survives a
    greyscale printer, which a hue-based scale would not.
    """
    t = max(0.0, min(1.0, score / 10.0))
    v = 0.80 - 0.72 * t
    return colors.Color(v, v, v)


def _bar(c, x, y, w, h, score: float) -> None:
    c.setFillColor(colors.HexColor("#eceef0"))
    c.rect(x, y, w, h, stroke=0, fill=1)
    c.setFillColor(_grey(score))
    c.rect(x, y, w * max(score, 0) / 10.0, h, stroke=0, fill=1)


def _skeleton(c, track: dict, x: float, y: float, w: float, h: float) -> None:
    """Draw one mid-walk frame from the stored keypoints."""
    if not track or not track.get("xy"):
        return
    names = track["landmarks"]
    idx = {n: i for i, n in enumerate(names)}
    frame = track["xy"][len(track["xy"]) // 2]
    vis = track["visibility"][len(track["visibility"]) // 2]

    xs = [p[0] for p in frame]
    ys = [p[1] for p in frame]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    scale = min(w / max(span_x, 1e-6), h / max(span_y, 1e-6)) * 0.88
    cx = x + w / 2 - (min(xs) + span_x / 2) * scale
    cy = y + h / 2 + (min(ys) + span_y / 2) * scale     # PDF y grows upward

    def pt(name):
        px, py = frame[idx[name]]
        return cx + px * scale, cy - py * scale

    c.setStrokeColor(INK)
    c.setLineWidth(1.6)
    c.setLineCap(1)
    for a, b in BONES:
        if idx.get(a) is None or idx.get(b) is None:
            continue
        if vis[idx[a]] < 0.4 or vis[idx[b]] < 0.4:
            continue
        ax, ay = pt(a)
        bx, by = pt(b)
        c.line(ax, ay, bx, by)
    c.setFillColor(INK)
    for n in names:
        if vis[idx[n]] >= 0.4:
            px, py = pt(n)
            c.circle(px, py, 1.5, stroke=0, fill=1)


def _header(c, title: str, subtitle: str) -> None:
    c.setFillColor(INK)
    c.setFont("Times-Roman", 22)
    c.drawString(MARGIN, PAGE_H - MARGIN - 6 * mm, title)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.5)
    c.drawString(MARGIN, PAGE_H - MARGIN - 12 * mm, subtitle)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.line(MARGIN, PAGE_H - MARGIN - 15 * mm,
           PAGE_W - MARGIN, PAGE_H - MARGIN - 15 * mm)


def _footer(c, page: int) -> None:
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.line(MARGIN, MARGIN + 10 * mm, PAGE_W - MARGIN, MARGIN + 10 * mm)
    c.setFillColor(FAINT)
    c.setFont("Helvetica", 7)
    c.drawString(MARGIN, MARGIN + 5 * mm,
                 "Runway AI - automated gait assessment")
    c.drawRightString(PAGE_W - MARGIN, MARGIN + 5 * mm, f"Page {page}")


def _wrap(c, text: str, x: float, y: float, width: float,
          leading: float = 4.4 * mm, font="Helvetica", size=9) -> float:
    """Draw wrapped text, returning the y below the last line."""
    c.setFont(font, size)
    words, line = text.split(), ""
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font, size) <= width:
            line = trial
        else:
            c.drawString(x, y, line)
            y -= leading
            line = word
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def build(session: dict, scores: dict, findings: list[dict],
          track: dict | None = None, user_name: str = "",
          quality: dict | None = None) -> bytes:
    """Render the assessment and return PDF bytes."""
    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=A4)
    c.setTitle("Runway performance assessment")

    when = str(session.get("recorded_at", ""))[:10]
    overall = float(session.get("overall_score") or 0)
    grade = session.get("grade") or ""

    # ---- page 1: the assessment ---------------------------------------- #
    _header(c, "Runway Performance Assessment",
            f"{user_name or 'Walk'} · {when} · {session.get('video_id', '')}")

    y = PAGE_H - MARGIN - 30 * mm
    c.setFillColor(INK)
    c.setFont("Times-Roman", 54)
    c.drawString(MARGIN, y, f"{overall:.1f}")
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawString(MARGIN + c.stringWidth(f"{overall:.1f}", "Times-Roman", 54) + 3 * mm,
                 y + 2 * mm, "/ 100")
    c.setFillColor(INK)
    c.setFont("Times-Italic", 15)
    c.drawString(MARGIN, y - 9 * mm, grade)

    if track:
        # Sits above the criterion table, which starts 26 mm below y - drawing
        # the caption at y-30 put it on top of the first rows.
        _skeleton(c, track, PAGE_W - MARGIN - 42 * mm, y - 14 * mm, 42 * mm, 38 * mm)
        c.setFillColor(FAINT)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(PAGE_W - MARGIN - 21 * mm, y - 19 * mm,
                            "tracked skeleton, mid-walk")

    # criterion table
    y -= 26 * mm
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(MARGIN, y, "CRITERION SCORES")
    y -= 6 * mm

    ordered = sorted(((k, v) for k, v in scores.items()), key=lambda kv: -kv[1])
    bar_x, bar_w = MARGIN + 62 * mm, 55 * mm
    for key, value in ordered:
        c.setFillColor(INK)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN, y, CRITERION_LABELS.get(key, key))
        _bar(c, bar_x, y - 0.6 * mm, bar_w, 2.6 * mm, value)
        c.setFont("Helvetica-Bold", 9)
        c.drawRightString(bar_x + bar_w + 14 * mm, y, f"{value:.1f}")
        c.setFillColor(FAINT)
        c.setFont("Helvetica", 7)
        c.drawRightString(PAGE_W - MARGIN, y,
                          f"w {CRITERION_WEIGHTS.get(key, 0):.2f}")
        y -= 6.2 * mm

    strong = [k for k, v in ordered if v >= 7.5]
    if strong:
        y -= 4 * mm
        c.setStrokeColor(RULE)
        c.line(MARGIN, y, PAGE_W - MARGIN, y)
        y -= 7 * mm
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(MARGIN, y, "WORKING WELL")
        y -= 5.5 * mm
        c.setFillColor(INK)
        _wrap(c, " · ".join(CRITERION_LABELS.get(k, k) for k in strong),
              MARGIN, y, PAGE_W - 2 * MARGIN, font="Helvetica", size=9)

    _footer(c, 1)
    c.showPage()

    # ---- page 2: what to work on --------------------------------------- #
    _header(c, "What to work on", f"{user_name or 'Walk'} · {when}")
    y = PAGE_H - MARGIN - 28 * mm

    # One row per criterion. Findings are raised per rubric *band*, so a single
    # weak criterion can occupy several slots - rhythm filled two of the top
    # five here - which crowds out advice the reader has not already been given.
    # The highest-severity band stands for its criterion.
    seen, top = set(), []
    for f in findings:
        if f["criterion"] in seen:
            continue
        seen.add(f["criterion"])
        top.append(f)
        if len(top) == 5:
            break
    if not top:
        c.setFillColor(MUTED)
        _wrap(c, "No corrective findings were raised for this walk.",
              MARGIN, y, PAGE_W - 2 * MARGIN)
    for i, f in enumerate(top, 1):
        c.setFillColor(FAINT)
        c.setFont("Times-Roman", 15)
        c.drawString(MARGIN, y, f"{i:02d}")
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 9.5)
        label = CRITERION_LABELS.get(f["criterion"], f["criterion"])
        c.drawString(MARGIN + 10 * mm, y,
                     f"{label} — {scores.get(f['criterion'], 0):.1f}/10")
        c.setFillColor(MUTED)
        y = _wrap(c, f.get("message") or "", MARGIN + 10 * mm, y - 5 * mm,
                  PAGE_W - 2 * MARGIN - 10 * mm)
        if f.get("feature"):
            c.setFillColor(FAINT)
            c.setFont("Courier", 7.5)
            c.drawString(MARGIN + 10 * mm, y,
                         f"{f['feature']}: measured {f['measured']:.4g}  "
                         f"target {f['ideal']:.4g}")
            y -= 5 * mm
        y -= 3 * mm

    # ---- recording quality, if the pipeline flagged any ------------------ #
    # These were raised at analysis time. A PDF handed to a coach without them
    # presents a score from a two-second clip exactly like one from a clean
    # recording.
    warnings = (quality or {}).get("warnings") or []
    if warnings:
        y -= 3 * mm
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(MARGIN, y, "RECORDING QUALITY")
        y -= 5 * mm
        for w in warnings[:4]:
            c.setFillColor(INK)
            c.setFont("Helvetica", 8)
            c.drawString(MARGIN, y, "!")
            y = _wrap(c, w, MARGIN + 4 * mm, y, PAGE_W - 2 * MARGIN - 4 * mm,
                      font="Helvetica", size=8)
            y -= 1 * mm
    if (quality or {}).get("coverage") in ("low", "moderate"):
        c.setFillColor(MUTED)
        y = _wrap(c,
                  f"Some measurements for this walk sit outside the range the "
                  f"models were trained on ({quality['coverage']} confidence), "
                  "so those scores are extrapolated.",
                  MARGIN, y, PAGE_W - 2 * MARGIN, font="Helvetica", size=8)
        y -= 2 * mm

    # ---- the caveat, on the page, not in a footnote --------------------- #
    y = max(y, MARGIN + 52 * mm)
    c.setStrokeColor(RULE)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 6 * mm
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(MARGIN, y, "HOW TO READ THIS")
    y -= 5 * mm
    c.setFillColor(MUTED)
    _wrap(c,
          "Scores come from twelve models trained on generated walks and "
          "validated against a single real video, with a mean out-of-fold error "
          "of about 0.9 points per criterion. Stride length, rhythm, arm swing, "
          "symmetry and hip movement are the least reliable: they are largely "
          "sagittal or need several gait cycles, and a front-on camera "
          "foreshortens them. Treat this as a structured second opinion "
          "alongside a coach, not as a measurement instrument.",
          MARGIN, y, PAGE_W - 2 * MARGIN, font="Helvetica", size=8)

    _footer(c, 2)
    c.save()
    return buf.getvalue()
