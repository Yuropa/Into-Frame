"""Generates a LaTeX-style PDF report from accumulated ReportSections."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 25 * mm
T_MARGIN = 30 * mm   # extra headroom; header drawn in top margin area on body pages
B_MARGIN = 25 * mm
CONTENT_W = PAGE_W - L_MARGIN - R_MARGIN

_DARK = HexColor("#1a1a2e")
_RULE = HexColor("#cccccc")
_GRAY = HexColor("#888888")
_CAPTION_GRAY = HexColor("#555555")

# ---------------------------------------------------------------------------
# Pipeline narrative — static overview text
# ---------------------------------------------------------------------------
_PIPELINE_OVERVIEW = (
    "Into Frame transforms a single input photograph into a fully explorable interactive "
    "3D environment using a multi-stage generative AI pipeline. The system begins by "
    "analysing the scene with a vision-language model, which produces a natural language "
    "caption used to guide panorama generation and other text-conditioned stages. "
    "A monocular depth model then estimates metric depth for every pixel, enabling "
    "accurate 3D understanding of the scene geometry."
    "\n\n"
    "From the depth map and original image, a full 360° equirectangular panorama is "
    "synthesised, extending the field of view to a complete spherical environment. "
    "The panorama is simultaneously analysed for lighting: an environment map is estimated "
    "and embedded in the scene for physically-based rendering. Where present, a skybox "
    "inpainting stage fills any gaps at the horizon and nadir of the panorama."
    "\n\n"
    "Ground-plane depth samples are projected into a top-down height grid. That grid is "
    "converted into a variable-density terrain mesh — densely tessellated near the camera "
    "and sparse at the horizon — then textured from a top-down bake of the panorama. "
    "Foreground objects are detected, segmented, and individually reconstructed as textured "
    "3D meshes using a single-image reconstruction model. All assets are assembled into a "
    "scene description that is streamed to the Unity or visionOS client for real-time "
    "interactive exploration."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_rl(pil_img: PILImage.Image, max_w: float, max_h: Optional[float] = None) -> RLImage:
    rgb = pil_img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "PNG")
    buf.seek(0)
    w, h = pil_img.size
    scale = max_w / w
    if max_h is not None:
        scale = min(scale, max_h / h)
    return RLImage(buf, width=w * scale, height=h * scale)


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "cover_title": ParagraphStyle(
            "cover_title", fontName="Times-Bold", fontSize=30, leading=36,
            alignment=TA_CENTER, spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontName="Times-Roman", fontSize=14, leading=20,
            alignment=TA_CENTER, textColor=HexColor("#444444"), spaceAfter=6,
        ),
        "cover_date": ParagraphStyle(
            "cover_date", fontName="Times-Italic", fontSize=12, leading=16,
            alignment=TA_CENTER, textColor=_GRAY, spaceAfter=24,
        ),
        "abstract_label": ParagraphStyle(
            "abstract_label", fontName="Times-Bold", fontSize=12, leading=16,
            alignment=TA_CENTER, spaceBefore=16, spaceAfter=6,
        ),
        "abstract": ParagraphStyle(
            "abstract", fontName="Times-Italic", fontSize=11, leading=16,
            alignment=TA_JUSTIFY, leftIndent=32, rightIndent=32, spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "section", fontName="Times-Bold", fontSize=14, leading=18,
            alignment=TA_LEFT, spaceBefore=0, spaceAfter=4, textColor=_DARK,
        ),
        "body": ParagraphStyle(
            "body", fontName="Times-Roman", fontSize=11, leading=16,
            alignment=TA_JUSTIFY, spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Times-Italic", fontSize=9, leading=12,
            alignment=TA_CENTER, textColor=_CAPTION_GRAY, spaceBefore=2, spaceAfter=10,
        ),
    }


def _footer_only(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(_GRAY)
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(_GRAY)
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, str(canvas.getPageNumber()))
    y_rule = PAGE_H - 20 * mm
    canvas.setStrokeColor(_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(L_MARGIN, y_rule, PAGE_W - R_MARGIN, y_rule)
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(_GRAY)
    canvas.drawString(L_MARGIN, y_rule + 2 * mm, "Into Frame")
    canvas.drawRightString(PAGE_W - R_MARGIN, y_rule + 2 * mm, "Pipeline Report")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Story builders
# ---------------------------------------------------------------------------

def _cover_story(st, caption: Optional[str], run_date: Optional[str],
                  input_image: Optional[PILImage.Image]) -> list:
    elems: list = []

    elems.append(Spacer(1, 56 * mm))
    elems.append(Paragraph("Into Frame", st["cover_title"]))
    elems.append(Paragraph("Pipeline Generation Report", st["cover_sub"]))
    elems.append(Paragraph(run_date or date.today().isoformat(), st["cover_date"]))
    elems.append(HRFlowable(
        width=CONTENT_W * 0.55, thickness=0.75, color=_RULE,
        spaceAfter=14, spaceBefore=4,
    ))

    if input_image is not None:
        elems.append(Spacer(1, 6 * mm))
        rl = _pil_to_rl(input_image, CONTENT_W * 0.72, 80 * mm)
        elems.append(Table(
            [[rl]], colWidths=[CONTENT_W],
            style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]),
        ))
        elems.append(Paragraph("Input photograph", st["caption"]))

    if caption:
        elems.append(Paragraph("Abstract", st["abstract_label"]))
        elems.append(Paragraph(caption, st["abstract"]))

    return elems


def _overview_story(st) -> list:
    elems: list = []
    elems.append(Spacer(1, 4 * mm))
    elems.append(Paragraph("Pipeline Overview", st["section"]))
    elems.append(HRFlowable(
        width=CONTENT_W, thickness=0.5, color=_RULE,
        spaceAfter=8, spaceBefore=2,
    ))
    for para in _PIPELINE_OVERVIEW.split("\n\n"):
        elems.append(Paragraph(para.strip(), st["body"]))
    return elems


def _section_story(st, section, section_number: int) -> list:
    elems: list = []

    elems.append(Spacer(1, 6 * mm))
    elems.append(Paragraph(f"{section_number}.&nbsp;&nbsp;{section.title}", st["section"]))
    elems.append(HRFlowable(
        width=CONTENT_W, thickness=0.5, color=_RULE,
        spaceAfter=8, spaceBefore=2,
    ))

    if section.body:
        for para in section.body.split("\n\n"):
            para = para.strip()
            if para:
                elems.append(Paragraph(para, st["body"]))

    if section.stats:
        col_a = CONTENT_W * 0.38
        col_b = CONTENT_W * 0.62
        rows = [
            [
                Paragraph(f"<b>{k}</b>", st["body"]),
                Paragraph(str(v), st["body"]),
            ]
            for k, v in section.stats.items()
        ]
        tbl = Table(rows, colWidths=[col_a, col_b])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Times-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("LEADING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, HexColor("#e8e8e8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elems.append(tbl)
        elems.append(Spacer(1, 8 * mm))

    if section.images:
        if len(section.images) == 1:
            pil_img, cap = section.images[0]
            rl = _pil_to_rl(pil_img, CONTENT_W, 120 * mm)
            elems.append(Table(
                [[rl]], colWidths=[CONTENT_W],
                style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]),
            ))
            elems.append(Paragraph(
                f"Figure&nbsp;{section_number}.1&nbsp;&#8212;&nbsp;{cap}", st["caption"],
            ))
        else:
            col_w = (CONTENT_W - 6 * mm) / 2
            fig_idx = 1
            for i in range(0, len(section.images), 2):
                left = section.images[i]
                right = section.images[i + 1] if i + 1 < len(section.images) else None

                pil_l, cap_l = left
                rl_l = _pil_to_rl(pil_l, col_w, 80 * mm)

                if right is not None:
                    pil_r, cap_r = right
                    rl_r = _pil_to_rl(pil_r, col_w, 80 * mm)
                    img_row = [rl_l, rl_r]
                    cap_row = [
                        Paragraph(f"Figure&nbsp;{section_number}.{fig_idx}&nbsp;&#8212;&nbsp;{cap_l}", st["caption"]),
                        Paragraph(f"Figure&nbsp;{section_number}.{fig_idx+1}&nbsp;&#8212;&nbsp;{cap_r}", st["caption"]),
                    ]
                    fig_idx += 2
                else:
                    img_row = [rl_l, Spacer(col_w, 1)]
                    cap_row = [
                        Paragraph(f"Figure&nbsp;{section_number}.{fig_idx}&nbsp;&#8212;&nbsp;{cap_l}", st["caption"]),
                        Paragraph("", st["caption"]),
                    ]
                    fig_idx += 1

                img_tbl = Table([img_row], colWidths=[col_w, col_w], spaceBefore=4)
                img_tbl.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]))
                cap_tbl = Table([cap_row], colWidths=[col_w, col_w])
                cap_tbl.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))

                elems.append(img_tbl)
                elems.append(cap_tbl)

    return elems


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(
    output_path: Path,
    sections: list,
    input_image: Optional[PILImage.Image] = None,
    caption: Optional[str] = None,
    run_date: Optional[str] = None,
) -> None:
    """Write a LaTeX-style PDF report to *output_path*."""
    st = _styles()
    story: list = []

    story.extend(_cover_story(st, caption, run_date, input_image))
    story.append(PageBreak())
    story.extend(_overview_story(st))

    for i, section in enumerate(sections):
        story.extend(_section_story(st, section, i + 1))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=L_MARGIN,
        rightMargin=R_MARGIN,
        topMargin=T_MARGIN,
        bottomMargin=B_MARGIN,
        title="Into Frame Pipeline Report",
        author="Into Frame",
    )
    doc.build(story, onFirstPage=_footer_only, onLaterPages=_header_footer)
