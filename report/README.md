# Into Frame — project report

First-pass LaTeX source for the ~20 page project report.

```
report/
├── main.tex          preamble, title block, bibliography — and the \input list
├── sections/         the actual body text, one file per section
│   ├── 00-abstract.tex
│   ├── 01-introduction.tex
│   ├── 02-related-work.tex
│   ├── 03-method-overview.tex
│   ├── 04-heightmap.tex
│   ├── 05-terrain.tex
│   ├── 06-texturing.tex
│   ├── 07-objects.tex
│   ├── 08-implementation.tex
│   ├── 09-results.tex        (longest — results, failure analysis, remediation)
│   └── 10-conclusions.tex
├── references.bib    42 entries, de-duplicated from ../resources.bib
├── figures/          diagram source + 3 data figures + 2 placeholders
└── README.md
```

**Edit the files in `sections/`, not `main.tex`.** Each is self-contained: it
opens with its own `\section`, closes no environment it did not open, and has
balanced braces, so you can reorder the `\input` list or comment out a section to
build a subset while drafting. Cross-references into a commented-out section warn
but still compile.

Numbering is `NN-` prefixed so the files sort in reading order; the numbers carry
no meaning to LaTeX, so renaming or renumbering only requires updating the
`\input` list.

## Building

```bash
latexmk -pdf main.tex
# or
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

I had no LaTeX toolchain available, so **this has not been compiled** — it is
validated statically only (balanced environments and braces, every `\cite` key
resolving against the bib, every `\ref` resolving to a `\label`, correct macro
arity). Expect to fix a small number of compile-time issues on the first run.

Packages used: `geometry graphicx booktabs amsmath amssymb siunitx microtype
caption subcaption enumitem xcolor hyperref cleveref tikz` (with the
`arrows.meta, positioning, fit, backgrounds` TikZ libraries).

The TikZ pipeline figure is the most likely thing to need a fix on first compile,
since it is the one part that could not be rendered here. Its layout *was*
verified — the matplotlib fallback shares the same coordinates and was rendered
and inspected — but the TikZ transcription itself is checked only structurally
(balanced delimiters, every node referenced is defined, every path terminated).

## Slides

`Into-Frame-slides.pptx` is a 21-slide 16:9 deck covering the same material, in
the same order as the report, closing on a references slide. It is
**generated**, not hand-edited:

```bash
python3 make_slides.py       # deps: python-pptx, Pillow, PyMuPDF
```

Edit `make_slides.py` and re-run; editing the `.pptx` directly means abandoning
the script. Copy is deliberately terse — the slide carries the claim and the
number, the speaker carries the sentence. Text lives in `build()`, the look lives in the palette and helpers
above it, and the pipeline diagram on slide 6 is drawn natively in PowerPoint
shapes rather than imported from `figures/`. Two figures are derived at build
time: `paris-terrain.pdf` is rasterised and split into its two panels, and the
ground-cover mask is cropped to its content.

Helvetica Neue has no arrow or subscript glyphs, so the deck spells arrows out
and uses baseline-shifted runs for subscripts. Do not reintroduce them.

There was previously a Beamer twin (`slides.tex` / `slides.pdf`); it was removed
when the deck was rebuilt, so the `.pptx` is now the only deck.

## Figures

Three are generated from the actual pipeline output in `samples.debug 2` and are
already in `figures/`:

| File | Content |
|---|---|
| `paris-terrain.pdf` | Paris height field before/after reconstruction, shared shading scale, the two invented hills annotated |
| `silhouette-composition.pdf` | Terrain share of the sky boundary per capture, with the 25% gate |
| `texel-density.pdf` | Median m/texel by distance ring, Rainier vs Shark Fin, log scale |

The pipeline diagram is **drawn in TikZ** and included directly by
`sections/03-method-overview.tex`:

| File | Role |
|---|---|
| `pipeline-overview.tex` | TikZ source — this is what the report renders |
| `make_pipeline_figure.py` | matplotlib fallback renderer, same coordinates |
| `pipeline-overview-fallback.pdf` | its output, if the TikZ won't compile |
| `pipeline-overview-preview.png` | what the layout looks like |
| `pipeline-overview.dot` | Graphviz version of the same graph |
| `pipeline-diagram-brief.md` | written spec of what the diagram says |

To swap to the image if the TikZ misbehaves, change one line in
`sections/03-method-overview.tex`:

```latex
\resizebox{\linewidth}{!}{\input{figures/pipeline-overview}}
% becomes
\includegraphics[width=\linewidth]{pipeline-overview-fallback}
```

Two are still **placeholders you need to supply** — the document compiles without
them, drawing a labelled frame instead, and picks each up automatically once the
file exists:

| Expected file | What it should show |
|---|---|
| `rainier-scene.jpg` | Mount Rainier reconstruction, viewed near the capture point |
| `sharkfin-scene.jpg` | Shark Fin Cove reconstruction, same |

Two more screenshots worth adding if you want to fill out the page count: a
before/after of the terrain UV-fold repair, and the Rainier meadow texture before
and after occluder-footprint inpainting.

## Notes

- `resources.bib` had two duplicate keys (`suvorov2021resolution`,
  `jain:hal-05617061`). `references.bib` is the de-duplicated copy — the original
  is untouched.
- `li2024svdquant` is in the bib but uncited. SVDQuant is not used in the
  pipeline; LTX-2 video generation has an optional quantisation path but not that
  one. Cite it or drop it.
- Several bib entries are arXiv-only where the work has since appeared at a
  venue. The report brief asks for the published venue, so these want checking
  before submission: `suvorov2021resolution` (LaMa, WACV 2022),
  `yu2023inpaint`, `zhang2023recognize` / `huang2023tag2text` / `huang2023open`
  (RAM/RAM++/Tag2Text), `liang2025luxdit`, `yang2024layerpano3d`, `lin2025dap`,
  `wu360anything`, `dreamcube_arxiv` (superseded by `dreamcube`, ICCV 2025 — the
  ICCV key is the one cited).
- All quantitative claims come from the `samples.debug 2` run
  (`pipeline-20260817-235917`) or from replaying stage code against its saved
  state. Anything you change in the pipeline invalidates the corresponding number.
