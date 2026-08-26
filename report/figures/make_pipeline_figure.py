#!/usr/bin/env python3
"""Render the pipeline overview figure with matplotlib.

This is the FALLBACK renderer. The figure the report actually uses is the TikZ
source in pipeline-overview.tex, which shares these exact coordinates and matches
the document's fonts. This script exists for two reasons:

  1. It was used to prototype and visually verify the layout before the TikZ was
     written, so the coordinates below are the ones that were checked.
  2. If the TikZ fails to compile, swapping the figure back to an image is a
     one-line change in sections/03-method-overview.tex:

         \\resizebox{\\linewidth}{!}{\\input{figures/pipeline-overview}}
     ->  \\includegraphics[width=\\linewidth]{pipeline-overview-fallback}

Usage:  python3 make_pipeline_figure.py
Writes: pipeline-overview-fallback.pdf, pipeline-overview-preview.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUT, BORD, LINE = "#0b0b0b", "#52514e", "#9a9a95", "#6b6b66"

LANES = [
    ("Terrain",      "#e8f0f9", "#c3d4e8", ["Height field", "Constrained\nreconstruction", "Mesh + texture"]),
    ("Objects",      "#fbeade", "#f0cdb8", ["Detection\n+ typing", "Appearance\nclustering", "3D assets"]),
    ("Ground cover", "#e4f3ec", "#bfe0d0", ["Spatial\nstatistics", "Population\nsynthesis"]),
    ("Water",        "#eae7f6", "#c9c3e4", ["Region\nextraction", "Levelled\nsurface"]),
    ("Sky",          "#f8f0d8", "#eddfae", ["Lighting\nestimation", "Skybox"]),
    ("Motion",       "#f0efeb", "#d8d6d0", ["Video\ngeneration", "Motion\nextraction", "Rigging"]),
]
BW, BH = 2.10, 0.84
LX0, LX1 = 5.55, 13.05
IN0, IN1 = LX0 + 0.30, LX1 - 0.30
LY = [6.30, 5.05, 3.80, 2.55, 1.30, -0.05]
PANX, PANW = 3.20, 2.24
AX, OX = 14.55, 17.60

fig, ax = plt.subplots(figsize=(15.6, 7.6))


def box(x, y, w, h, label, fc, ec=BORD, lw=0.9, fs=9.0, bold=False, z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 1,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(p, q, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=9, lw=0.85, color=LINE, zorder=2,
                                 linestyle=(0, (3.5, 2.5)) if dashed else "solid",
                                 shrinkA=0, shrinkB=0))


def centres(n):
    """Distribute n boxes evenly across the lane's inner span, so every lane is
    flush right regardless of how many stages it has."""
    if n == 1:
        return [IN0 + BW / 2]
    lo, hi = IN0 + BW / 2, IN1 - BW / 2
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


box(0.80, 3.10, 1.80, 1.05, "Single\nphotograph", "#eceae4", ec="#3a3a36", lw=1.5, bold=True)

ax.add_patch(FancyBboxPatch((PANX - PANW / 2, 1.72), PANW, 2.80,
                            boxstyle="round,pad=0.02,rounding_size=0.14",
                            fc="#faf9f5", ec=BORD, lw=1.1, zorder=1))
ax.text(PANX, 4.62, "Three panorama variants", ha="center", va="bottom",
        fontsize=8.8, color=MUT, style="italic")
PY = {"orig": 3.98, "terr": 3.06, "sky": 2.14}
for label, key in [("Original\npanorama", "orig"),
                   ("Object-removed\npanorama", "terr"),
                   ("Sky-inpainted\npanorama", "sky")]:
    box(PANX, PY[key], PANW - 0.22, 0.74, label, "#f4f2ec", fs=8.4)
arrow((1.70, 3.10), (PANX - PANW / 2 - 0.06, 3.10))

RIGHT = PANX + PANW / 2
src = {k: (RIGHT, v) for k, v in PY.items()}
# which panorama variant feeds which lane -- the load-bearing routing detail
feed = {0: ("terr", 0.0), 1: ("orig", 0.0), 2: ("orig", 0.10),
        3: ("terr", -0.10), 4: ("sky", 0.0), 5: ("orig", 0.16)}

for i, ((name, fc, ec, steps), y) in enumerate(zip(LANES, LY)):
    ax.add_patch(FancyBboxPatch((LX0, y - 0.63), LX1 - LX0, 1.26,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.0, zorder=1))
    ax.text(LX0 + 0.16, y + 0.44, name, ha="left", va="center",
            fontsize=8.8, color=MUT, fontweight="bold", zorder=4)
    cs = centres(len(steps))
    for j, s in enumerate(steps):
        box(cs[j], y - 0.05, BW, BH, s, "white", fs=8.4)
        if j:
            arrow((cs[j - 1] + BW / 2, y - 0.05), (cs[j] - BW / 2, y - 0.05))
    key, rad = feed[i]
    arrow(src[key], (LX0 - 0.05, y), rad=rad)
    arrow((LX1 + 0.05, y), (AX - 1.00, 3.10), rad=0.05 if y > 3.10 else -0.05)

cs3, cs2 = centres(3), centres(2)
arrow((cs3[1], LY[1] - 0.05 - BH / 2), (cs2[0], LY[2] - 0.05 + BH / 2), dashed=True, rad=-0.15)
arrow((RIGHT, PY["orig"]), (cs3[1], LY[0] - 0.05 + BH / 2), dashed=True, rad=-0.30)

box(AX, 3.10, 1.90, 1.10, "Scene\nassembly", "#eceae4", ec="#3a3a36", lw=1.4, bold=True)
arrow((AX + 0.95, 3.10), (OX - 1.20, 3.10))
box(OX, 3.10, 2.40, 1.15, "Explorable 3D\nenvironment", "#dfdcd3", ec="#3a3a36", lw=1.9, bold=True)

ax.plot([0.35, 1.05], [-1.30, -1.30], color=LINE, lw=0.85)
ax.text(1.17, -1.30, "geometry / assets", fontsize=8.3, va="center", color=MUT)
ax.plot([3.55, 4.25], [-1.30, -1.30], color=LINE, lw=0.85, linestyle=(0, (3.5, 2.5)))
ax.text(4.37, -1.30, "semantics", fontsize=8.3, va="center", color=MUT)

ax.set_xlim(-0.45, 19.05)
ax.set_ylim(-1.85, 7.15)
ax.axis("off")
plt.tight_layout()
plt.savefig("pipeline-overview-fallback.pdf", bbox_inches="tight")
plt.savefig("pipeline-overview-preview.png", bbox_inches="tight", dpi=110)
print("wrote pipeline-overview-fallback.pdf, pipeline-overview-preview.png")
