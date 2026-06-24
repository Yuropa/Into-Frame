"""Generates a LaTeX-style PDF report from accumulated ReportSections."""

from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image as RLImage,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 25 * mm
T_MARGIN = 30 * mm   # content starts 30 mm from top; header is drawn in the 10 mm gap
B_MARGIN = 25 * mm
CONTENT_W = PAGE_W - L_MARGIN - R_MARGIN
CONTENT_H = PAGE_H - T_MARGIN - B_MARGIN

_DARK = HexColor("#1a1a2e")
_ACCENT = HexColor("#3f51b5")
_RULE = HexColor("#cccccc")
_GRAY = HexColor("#888888")
_CAPTION_GRAY = HexColor("#555555")

# ---------------------------------------------------------------------------
# Pipeline narrative
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
    "The panorama is analysed for lighting using a diffusion-based illumination model, "
    "producing an environment map used for physically-based rendering of all assets. "
    "Where required, sky and horizon regions are inpainted to produce a seamless skybox."
    "\n\n"
    "Ground-plane depth samples are projected into a top-down height grid, which is "
    "converted into a variable-density terrain mesh — densely tessellated near the "
    "camera and sparse at the horizon. A texture baked from the panorama is applied to "
    "the terrain surface. Foreground objects are detected, segmented, and individually "
    "reconstructed as textured 3D meshes. All assets are assembled into a scene "
    "description streamed to the Unity or visionOS client for real-time exploration."
)


# ---------------------------------------------------------------------------
# Pipeline flow diagram
# ---------------------------------------------------------------------------

def _render_pipeline_diagram(section_titles: list[str]) -> Optional[PILImage.Image]:
    """Render a vertical flowchart of named pipeline stages using matplotlib."""
    if not section_titles:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = len(section_titles)
    box_h = 0.38
    gap = 0.18
    pad_x = 0.5
    pad_y = 0.4
    box_w = 5.2
    fig_w = box_w + 2 * pad_x
    fig_h = n * box_h + (n - 1) * gap + 2 * pad_y

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y_top = fig_h - pad_y

    for i, title in enumerate(section_titles):
        y = y_top - i * (box_h + gap) - box_h
        cx = pad_x + box_w / 2

        # Stage box
        rect = mpatches.FancyBboxPatch(
            (pad_x, y), box_w, box_h,
            boxstyle="round,pad=0.03",
            facecolor="#eef0fb",
            edgecolor="#3f51b5",
            linewidth=0.7,
            zorder=2,
        )
        ax.add_patch(rect)

        # Stage number badge
        badge = plt.Circle((pad_x + 0.28, y + box_h / 2), 0.13, color="#3f51b5", zorder=3)
        ax.add_patch(badge)
        ax.text(
            pad_x + 0.28, y + box_h / 2, str(i + 1),
            ha="center", va="center", fontsize=5.5,
            color="white", fontweight="bold", zorder=4,
        )

        # Title text
        ax.text(
            pad_x + 0.6, y + box_h / 2, title,
            ha="left", va="center", fontsize=7.5,
            fontfamily="serif", color="#1a1a2e", zorder=3,
        )

        # Arrow to next box
        if i < n - 1:
            ax.annotate(
                "",
                xy=(cx, y - gap + 0.01),
                xytext=(cx, y - 0.01),
                arrowprops=dict(
                    arrowstyle="->", color="#3f51b5", lw=0.7, mutation_scale=7,
                ),
                zorder=2,
            )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

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
        "toc_title": ParagraphStyle(
            "toc_title", fontName="Times-Bold", fontSize=16, leading=20,
            alignment=TA_LEFT, spaceAfter=10, textColor=_DARK,
        ),
        # "section" is also the TOC trigger name — keep it stable
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
        "diagram_caption": ParagraphStyle(
            "diagram_caption", fontName="Times-Italic", fontSize=9, leading=12,
            alignment=TA_CENTER, textColor=_CAPTION_GRAY, spaceBefore=2, spaceAfter=6,
        ),
    }


# ---------------------------------------------------------------------------
# Page callbacks
# ---------------------------------------------------------------------------

def _footer_only(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(_GRAY)
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def _header_footer(canvas, doc):
    canvas.saveState()
    # Page number
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(_GRAY)
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, str(canvas.getPageNumber()))
    # Header rule and text
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
# Doc class with TOC wiring
# ---------------------------------------------------------------------------

class _ReportDoc(BaseDocTemplate):
    """BaseDocTemplate subclass that notifies the TOC whenever a section heading is laid out."""

    def __init__(self, filename: str, toc: TableOfContents, **kwargs):
        BaseDocTemplate.__init__(self, filename, **kwargs)
        self._toc = toc

        frame = Frame(
            L_MARGIN, B_MARGIN, CONTENT_W, CONTENT_H,
            id="normal", showBoundary=0,
        )
        self.addPageTemplates([
            PageTemplate(id="first", frames=[frame], onPage=_footer_only),
            PageTemplate(id="normal", frames=[frame], onPage=_header_footer),
        ])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "section":
            raw = flowable.getPlainText()
            # Strip leading "N.  " section numbers and normalise whitespace
            text = re.sub(r"^\d+\.\s+", "", re.sub(r"\s+", " ", raw)).strip()
            self.notify("TOCEntry", (0, text, self.page, None))


# ---------------------------------------------------------------------------
# Story builders
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


def _cover_story(st, caption, run_date, input_image) -> list:
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
        rl = _pil_to_rl(input_image, CONTENT_W * 0.72, 78 * mm)
        elems.append(Table(
            [[rl]], colWidths=[CONTENT_W],
            style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]),
        ))
        elems.append(Paragraph("Input photograph", st["caption"]))
    if caption:
        elems.append(Paragraph("Abstract", st["abstract_label"]))
        elems.append(Paragraph(caption, st["abstract"]))
    return elems


def _toc_page_story(st, toc: TableOfContents) -> list:
    elems: list = []
    elems.append(Spacer(1, 6 * mm))
    elems.append(Paragraph("Contents", st["toc_title"]))
    elems.append(HRFlowable(
        width=CONTENT_W, thickness=0.5, color=_RULE,
        spaceAfter=10, spaceBefore=2,
    ))
    elems.append(toc)
    return elems


def _overview_story(st, section_titles: list[str]) -> list:
    elems: list = []
    elems.append(Spacer(1, 4 * mm))
    # Use the "section" style so it appears in the TOC
    elems.append(Paragraph("Pipeline Overview", st["section"]))
    elems.append(HRFlowable(
        width=CONTENT_W, thickness=0.5, color=_RULE,
        spaceAfter=8, spaceBefore=2,
    ))
    for para in _PIPELINE_OVERVIEW.split("\n\n"):
        elems.append(Paragraph(para.strip(), st["body"]))

    diagram = _render_pipeline_diagram(section_titles)
    if diagram is not None:
        elems.append(Spacer(1, 4 * mm))
        # Fit the diagram in at most half the content height so it stays on one page
        rl = _pil_to_rl(diagram, CONTENT_W * 0.5, CONTENT_H * 0.52)
        elems.append(Table(
            [[rl]], colWidths=[CONTENT_W],
            style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]),
        ))
        elems.append(Paragraph(
            "Figure&nbsp;0.1&nbsp;—&nbsp;Pipeline stage sequence",
            st["diagram_caption"],
        ))
    return elems


def _section_story(st, section, section_number: int) -> list:
    elems: list = []
    elems.append(Spacer(1, 6 * mm))
    # Section heading — "section" style triggers TOC entry via afterFlowable
    elems.append(Paragraph(
        f"{section_number}.  {section.title}", st["section"],
    ))
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
            [Paragraph(f"<b>{k}</b>", st["body"]), Paragraph(str(v), st["body"])]
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
            rl = _pil_to_rl(pil_img, CONTENT_W, 115 * mm)
            elems.append(Table(
                [[rl]], colWidths=[CONTENT_W],
                style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]),
            ))
            elems.append(Paragraph(
                f"Figure {section_number}.1 — {cap}", st["caption"],
            ))
        else:
            col_w = (CONTENT_W - 6 * mm) / 2
            fig_idx = 1
            for i in range(0, len(section.images), 2):
                left = section.images[i]
                right = section.images[i + 1] if i + 1 < len(section.images) else None

                pil_l, cap_l = left
                rl_l = _pil_to_rl(pil_l, col_w, 75 * mm)

                if right is not None:
                    pil_r, cap_r = right
                    rl_r = _pil_to_rl(pil_r, col_w, 75 * mm)
                    img_row = [rl_l, rl_r]
                    cap_row = [
                        Paragraph(f"Figure {section_number}.{fig_idx} — {cap_l}", st["caption"]),
                        Paragraph(f"Figure {section_number}.{fig_idx+1} — {cap_r}", st["caption"]),
                    ]
                    fig_idx += 2
                else:
                    img_row = [rl_l, Spacer(col_w, 1)]
                    cap_row = [
                        Paragraph(f"Figure {section_number}.{fig_idx} — {cap_l}", st["caption"]),
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

    # Build the TOC flowable — entries are populated during multiBuild
    toc = TableOfContents()
    toc.dotsMinLevel = 0
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1", fontName="Times-Roman", fontSize=11, leading=16,
            leftIndent=0, spaceAfter=3, rightIndent=12,
        ),
    ]

    section_titles = [s.title for s in sections]

    story: list = []

    # Cover page (uses 'first' template — no header rule)
    story.extend(_cover_story(st, caption, run_date, input_image))
    story.append(NextPageTemplate("normal"))
    story.append(PageBreak())

    # Table of contents page
    story.extend(_toc_page_story(st, toc))
    story.append(PageBreak())

    # Pipeline overview (with diagram)
    story.extend(_overview_story(st, section_titles))

    # Per-stage sections
    for i, section in enumerate(sections):
        story.extend(_section_story(st, section, i + 1))

    doc = _ReportDoc(
        str(output_path),
        toc=toc,
        pagesize=A4,
        leftMargin=L_MARGIN,
        rightMargin=R_MARGIN,
        topMargin=T_MARGIN,
        bottomMargin=B_MARGIN,
        title="Into Frame Pipeline Report",
        author="Into Frame",
    )
    # multiBuild does two layout passes so TOC page numbers are accurate
    doc.multiBuild(story)
