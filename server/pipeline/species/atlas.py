"""Part textures for the species meshes: a flower head, a leaf, a conifer spray,
generated for the species the plan names, matted, recoloured to the bucket's
measured palette and packed into one atlas per species.

The procedural plants (blender_species.py) were flat-coloured polygons, one
swatch per face: a paintbrush head was a bundle of maroon strips, a fir branch
two green strips, and the judge called them "low-poly blobs, flat-shaded".
Game vegetation is textured cards on a procedural skeleton, and that is what
this makes: FLUX.1-dev paints each part of the species isolated on a plain
background from the plan's words (its Latin name, its phrase, its colours),
BiRefNet cuts it out, the cutout's mean colour is shifted to the colour the
bucket's own crops measured (the plant keeps the photograph's colour, the
texture keeps its detail), and the parts go into a 1024x1024 atlas with a
corner left for the builder's stem swatches.

Atlas layout (pixel rows top-down): parts in 512x512 cells, cell 0 top-left,
1 top-right, 2 bottom-left; the builder's 64x64 palette at the bottom-right
cell's lower-left corner (PALETTE_ORIGIN). Each part's uv rect covers exactly
its resized cutout, so a card keeps the picture's own aspect.

A flower head is painted from the photograph's own specimen when the Species
Director boxed one (species["head_box"]) and from its words for it
(species["look"]). From the name alone FLUX painted its idea of the species:
Rainier's magenta paintbrush was a tight pink cone of buds and the pasqueflower's
seed head a white dandelion disc, and the meadow stood full of flowers that are
not in the photograph. The boxed head is cut from the photograph, matted onto
white and given to FLUX.1-Redux, whose image tokens are appended to the encoded
prompt at reference_scale: the painting is a clean, sharp, whole version of the
head that was photographed, a few dozen pixels of it being enough to say what
it is.

Cached across runs by the species, its palette, the prompt and the reference
under cache/species/atlas_<hash>/ (atlas.png + parts.json).
"""
from __future__ import annotations

import hashlib
import json
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image as PILImage
from skimage import color as skcolor

REDUX_MODEL_ID = "black-forest-labs/FLUX.1-Redux-dev"
ATLAS_PX = 1024
CELL_PX = 512
PALETTE_PX = 64
PALETTE_ORIGIN = (CELL_PX, ATLAS_PX - PALETTE_PX)   # (x, y) in pixel rows top-down
_CELLS = [(0, 0), (CELL_PX, 0), (0, CELL_PX)]

# Parts per form, the palette part each is recoloured to, and the prompt. {name},
# {phrase} and {colours} come from the plan's species entry. Every prompt asks for
# the part isolated on plain white so the matte is clean, upright with its base at
# the bottom so the card's up axis is the image's.
PARTS: dict[str, list[tuple[str, str, str]]] = {
    "spike": [
        # Shape words are left to the species ("full and bushy" made the lousewort a
        # dandelion; "much taller than wide" made the paintbrush a sliver).
        ("head", "bloom", "the whole flower spike of {name} ({phrase}) seen from the side, {colours}, in its "
                          "true proportions, its bracts and florets packed close around the stem, every bract in "
                          "sharp focus, upright with its base at the bottom, isolated on a plain white "
                          "background, macro photograph, soft even daylight, no stem below the spike"),
        ("leaf", "foliage", "a single green leaf of a wild alpine meadow plant, about three times longer than wide, "
                            "upright with its base at the bottom, isolated on a plain white background, photograph, "
                            "sharp focus, soft even daylight"),
    ],
    "raceme": [
        ("head", "bloom", "a single {phrase}, the flower raceme of {name}, {colours}, whorls of pea flowers up a "
                          "vertical stalk, base at the bottom, isolated on a plain white background, macro "
                          "photograph, soft even daylight"),
        ("leaf", "foliage", "a single palmate green leaf of {name} seen from above, leaflets radiating from the "
                            "centre, isolated on a plain white background, photograph, sharp focus"),
    ],
    "seedhead": [
        ("head", "bloom", "a single {phrase} of {name}, a silky fuzzy seed head on the tip of its stalk, {colours}, "
                          "upright with the stalk at the bottom, isolated on a plain white background, macro "
                          "photograph, soft backlight through the hairs"),
        ("leaf", "foliage", "a single finely divided green leaf of an alpine meadow plant, about twice as long as "
                            "wide, upright, isolated on a plain white background, photograph, sharp focus"),
    ],
    "conifer": [
        ("spray", "foliage", "a single branch spray of {name}, {phrase}, dense short {colours} needles on a twig, the "
                             "twig's base at the bottom and the tip pointing up, flat frontal view, isolated on a "
                             "plain white background, photograph, sharp focus, soft even daylight"),
    ],
    # A broadleaf tree's crown is filled with this one spray. The plan's phrase is the
    # whole tree ("green plane tree") and is left out: with it the painting is a tree.
    # Asked for "a branch tip", FLUX painted one compound leaf on a pale stalk and the
    # crown was drawn in stalks; a bough's rounded mass of leaves hides its twig.
    "broadleaf": [
        ("spray", "foliage", "a cut bough of {name} foliage, a full rounded mass of overlapping {colours} leaves, "
                             "as wide as it is tall, no bare stem, flat frontal view, isolated on a plain white "
                             "background, photograph, sharp focus, soft even daylight"),
    ],
    # A weeping tree: the spray caps its dome, the strand is a hanging curtain of
    # branchlets (painted hanging; the atlas stores it turned over, its attachment
    # at the bottom like every other part, and the builder hangs it by that end).
    "weeping": [
        ("spray", "foliage", "a leafy branch tip of {name}, a dense spray of many narrow {colours} leaves "
                             "overlapping and hiding the twig, the foliage filling the frame, flat frontal view, "
                             "isolated on a plain white background, photograph, sharp focus, soft even daylight"),
        ("strand", "foliage", "a bunch of several long thin {name} branchlets hanging side by side like a curtain, "
                              "each covered in narrow {colours} leaves along its whole length, about three times "
                              "longer than wide, isolated on a plain white background, photograph, sharp focus, "
                              "soft even daylight"),
    ],
    "tuft": [
        ("blade", "foliage", "a clump of {phrase}, {name}, slender green grass blades rising from one base at the "
                             "bottom, isolated on a plain white background, photograph, sharp focus"),
    ],
    "shrub": [
        ("leaf", "foliage", "a single leafy twig of {name}, {phrase}, {colours}, several small leaves on a short "
                            "stem with its base at the bottom, isolated on a plain white background, photograph, "
                            "sharp focus, soft even daylight"),
    ],
}


def decontaminate(rgba: np.ndarray, cutoff: int = 128, rim: int = 2, reach: float = 4.0,
                  thin_reach: float = 16.0, backdrop=(255.0, 255.0, 255.0)) -> np.ndarray:
    """The matte's outermost pixels without the backdrop that is mixed into them.
    The part is painted on white and the matte is cut along the painting's own
    antialiased edge, so the rim of every part is the part mixed with white: a
    fir's spray came out [67, 83, 50] inside and [112-137, 123-143, 101-129] over
    the eighth of its pixels along the edge, a pale outline round every needle
    that the far mips then spread over the whole card.

    A rim pixel within `reach` of the inside (the matte eroded by `rim`) takes the
    nearest inside colour. A thinner feature (a needle's tip, a stem) within
    `thin_reach` of the inside keeps its own colour less its share of the
    backdrop, measured along the line from the nearest inside colour to white.
    A part with no inside at all (a blade of grass) is left as painted."""
    from scipy import ndimage
    opaque = rgba[..., 3] >= cutoff
    inner = ndimage.binary_erosion(opaque, iterations=rim)
    if not inner.any():
        return rgba
    dist, idx = ndimage.distance_transform_edt(~inner, return_indices=True)
    inside = rgba[..., :3][idx[0], idx[1]].astype(np.float32)
    out = rgba.copy()
    take = opaque & ~inner & (dist <= reach)
    out[..., :3][take] = inside[take].astype(np.uint8)
    thin = opaque & ~inner & (dist > reach) & (dist <= thin_reach)
    if thin.any():
        own = rgba[..., :3][thin].astype(np.float32)
        to_white = np.asarray(backdrop, dtype=np.float32) - inside[thin]
        share = np.sum((own - inside[thin]) * to_white, axis=1) / np.maximum(np.sum(to_white * to_white, axis=1), 1.0)
        share = np.clip(share, 0.0, 1.0)[:, None]
        out[..., :3][thin] = np.clip(np.rint(own - share * to_white), 0, 255).astype(np.uint8)
    return out


def alpha_bleed(rgba: np.ndarray, cutoff: int = 128, near: float = 2.0, far: float = 14.0) -> np.ndarray:
    """Every transparent pixel takes the colour of the nearest opaque one. The
    matte leaves the background black under alpha 0, and the renderer's bilinear
    filter (and every resize) mixes that black into the edge: an ink line around
    every leaf and head, the "toon shading" of band24.

    Beyond `near` pixels of the matte the nearest colour gives way, by `far`, to
    the colours around it blurred: the nearest opaque pixel of an empty half
    atlas is one speck of stem, and a mip that averages a card with its empty
    surroundings took that speck's colour over the part's own (Rainier's far firs
    drawn grey-lilac from a green spray)."""
    from scipy import ndimage
    a = rgba[..., 3]
    opaque = a >= cutoff
    if not opaque.any() or opaque.all():
        return rgba
    dist, idx = ndimage.distance_transform_edt(~opaque, return_indices=True)
    out = rgba.copy()
    nearest = rgba[..., :3][idx[0], idx[1]].astype(np.float32)
    # The opaque colours spread by a normalised blur: the local mean of the parts.
    w = ndimage.gaussian_filter(opaque.astype(np.float32), 24.0)
    spread = np.stack([ndimage.gaussian_filter(rgba[..., c].astype(np.float32) * opaque, 24.0) for c in range(3)], axis=-1)
    known = w > 1e-3
    spread = np.where(known[..., None], spread / np.maximum(w, 1e-3)[..., None], nearest)
    # Further out than the blur reaches: the nearest place it does reach.
    if not known.all():
        far_idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        spread = spread[far_idx[0], far_idx[1]]
    t = np.clip((dist - near) / max(far - near, 1e-6), 0.0, 1.0)[..., None].astype(np.float32)
    filled = nearest * (1.0 - t) + spread * t
    out[..., :3] = np.where(opaque[..., None], rgba[..., :3], np.clip(np.rint(filled), 0, 255).astype(np.uint8))
    return out


def outline_profile(alpha: np.ndarray, bands: int = 8, cutoff: int = 128) -> list[float]:
    """The cutout's width in `bands` bands from its base to its tip, each as a share
    of the cutout's whole width (the widest opaque run of the band's rows, median
    over the rows): the outline the plant builder turns into the head's body."""
    opaque = np.asarray(alpha) >= cutoff
    h, w = opaque.shape
    out = []
    for k in range(bands):
        r1, r0 = h - (k * h) // bands, h - ((k + 1) * h) // bands      # base first
        rows = opaque[r0:max(r1, r0 + 1)]
        spans = [(np.flatnonzero(r)[-1] - np.flatnonzero(r)[0] + 1) / max(w, 1) for r in rows if r.any()]
        out.append(round(float(np.median(spans)) if spans else 0.0, 3))
    return out


def recolour(rgba: np.ndarray, target_rgb, strength: float = 0.85, lightness: float = 0.5,
             chroma: float = 1.0) -> np.ndarray:
    """Shift the opaque pixels' mean colour (Lab) to the target: the whole of the
    chroma, `lightness` of the lightness difference, each scaled by strength.
    Per-pixel variation (the texture) is kept. `chroma` < 1 then pulls every
    opaque pixel toward grey by that factor: a painting lit again by the scene's
    sun reads oversaturated (band26's "oversaturated yellow spikes")."""
    out = rgba.astype(np.float32).copy()
    alpha = out[..., 3] > 64
    if alpha.sum() < 16:
        return rgba
    rgb = out[..., :3] / 255.0
    lab = skcolor.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(rgb.shape)
    mean = lab[alpha].mean(axis=0)
    tgt = skcolor.rgb2lab(np.asarray(target_rgb, dtype=np.float64).reshape(1, 1, 3) / 255.0).reshape(3)
    delta = tgt - mean
    delta[0] *= lightness
    lab[alpha] += strength * delta
    if chroma != 1.0:
        lab[alpha, 1:] *= chroma
    shifted = np.clip(skcolor.lab2rgb(lab.reshape(-1, 1, 3)).reshape(rgb.shape), 0.0, 1.0)
    out[..., :3] = shifted * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _colour_words(species: dict) -> str:
    words = [str(w) for w in (species.get("colours") or []) if isinstance(w, str)]
    return ", ".join(words) if words else "natural colours"


# A head the Species Director described: its words lead, the name follows.
LOOK_HEAD_PROMPT = ("{look}. One flower head of {name}, whole and alone, upright with its base at the bottom, "
                    "every part in sharp focus, isolated on a plain white background, macro photograph, soft "
                    "even daylight, no stem below the head")


def part_prompts(form: str, species: dict) -> list[tuple[str, str, str]]:
    name = str(species.get("name") or species.get("phrase") or "the plant")
    phrase = str(species.get("phrase") or name)
    fields = {"name": name, "phrase": phrase, "colours": _colour_words(species)}
    look = " ".join(str(species.get("look") or "").split()).rstrip(".")
    out = []
    for part, pal, prompt in PARTS.get(form, PARTS["spike"]):
        if part == "head" and look:
            prompt = LOOK_HEAD_PROMPT.replace("{look}", look.replace("{", "(").replace("}", ")"))
        out.append((part, pal, prompt.format(**fields)))
    return out


def head_reference(photo: Optional[PILImage.Image], species: dict, margin: float = 0.15,
                   min_px: int = 12) -> Optional[PILImage.Image]:
    """The photographed head the Species Director boxed (species["head_box"], [x, y, w, h]
    as fractions), with a margin; None without a box or when it is under min_px a side."""
    box = species.get("head_box")
    if photo is None or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    W, H = photo.size
    x, y, w, h = (float(v) for v in box)
    x0, y0 = max(0, int((x - margin * w) * W)), max(0, int((y - margin * h) * H))
    x1, y1 = min(W, int(round((x + w + margin * w) * W))), min(H, int(round((y + h + margin * h) * H)))
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return None
    return photo.convert("RGB").crop((x0, y0, x1, y1))


class SpeciesAtlas:
    """FLUX part images -> BiRefNet mattes -> recoloured cutouts -> one atlas per species."""

    def __init__(self, device, torch_dtype: Any, cache_dir: Path, log: Logger,
                 image_px: int = 768, steps: int = 28, guidance: float = 3.5, seed: int = 0,
                 reference_scale: float = 0.6, reference_matte: bool = True):
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = Path(cache_dir)
        self.log = log
        self.image_px = int(image_px)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.seed = int(seed)
        # How strongly the photographed head steers its painting (the Redux tokens'
        # scale beside the text's; 0 turns the reference off), and whether the
        # reference is matted onto white first.
        self.reference_scale = float(reference_scale)
        self.reference_matte = bool(reference_matte)
        self._flux = None
        self._matter = None
        self._redux = None

    # ── models, loaded on first use ─────────────────────────────────────────
    def _painter(self):
        if self._flux is None:
            from pipeline.inpainting.inpainting_flux import InPaintingFlux, DEV_MODEL_ID
            self._flux = InPaintingFlux(self.device, self.torch_dtype, base_model=DEV_MODEL_ID)
        return self._flux

    def _seg(self):
        if self._matter is None:
            from pipeline.segmentation.foreground_segmentation import ForegroundSeg
            self._matter = ForegroundSeg(self.device)
        return self._matter

    def _reference_encoder(self):
        """(SigLIP processor, SigLIP vision model, Redux embedder) on the device."""
        if self._redux is None:
            from transformers import SiglipImageProcessor, SiglipVisionModel
            from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
            processor = SiglipImageProcessor.from_pretrained(REDUX_MODEL_ID, subfolder="feature_extractor")
            vision = SiglipVisionModel.from_pretrained(REDUX_MODEL_ID, subfolder="image_encoder",
                                                       torch_dtype=self.torch_dtype).to(self.device)
            embedder = ReduxImageEncoder.from_pretrained(REDUX_MODEL_ID, subfolder="image_embedder",
                                                         torch_dtype=self.torch_dtype).to(self.device)
            self._redux = (processor, vision, embedder)
        return self._redux

    def reference_image(self, crop: PILImage.Image, work: Optional[Path] = None, tag: str = "ref",
                        px: int = 384) -> PILImage.Image:
        """The photographed head as the painter sees it: enlarged, matted onto white
        when the matte finds a head in it (between 8% and 85% of the crop), centred
        on a white square. A slender spike is not squashed to the encoder's square."""
        from util.image_utils import Image
        scale = (px - 16) / max(crop.size)
        img = crop.convert("RGB").resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                                         PILImage.LANCZOS)
        if self.reference_matte:
            try:
                seg = self._seg().segment(Image(img))
                alpha = np.asarray(seg.masks[0], dtype=np.float32)
                share = float((alpha > 0.5).mean())
                if 0.08 <= share <= 0.85:
                    rgb = np.asarray(img, dtype=np.float32)
                    img = PILImage.fromarray((rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]))
                                             .clip(0, 255).astype(np.uint8), "RGB")
                else:
                    self.log.info(f"  atlas: the matte holds {share:.0%} of the photographed head's crop; "
                                  f"the crop is the reference as it is")
            except Exception as exc:
                self.log.warning(f"  atlas: could not matte the reference ({exc}); the crop is the reference as it is")
        square = PILImage.new("RGB", (px, px), (255, 255, 255))
        square.paste(img, ((px - img.width) // 2, (px - img.height) // 2))
        if work is not None:
            square.save(Path(work) / f"{tag}_reference.png")
        return square

    def _reference_embeds(self, prompt: str, reference: PILImage.Image, scale: float):
        """The encoded prompt with the reference's Redux tokens appended, scaled."""
        import torch
        pipe = self._painter().pipeline
        processor, vision, embedder = self._reference_encoder()
        with torch.no_grad():
            prompt_embeds, pooled, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device=self.device,
                                                          max_sequence_length=512)
            pixels = processor(images=reference, return_tensors="pt").pixel_values.to(self.device, self.torch_dtype)
            tokens = embedder(vision(pixels).last_hidden_state).image_embeds
        tokens = tokens.to(prompt_embeds.device, prompt_embeds.dtype) * float(scale)
        return torch.cat([prompt_embeds, tokens], dim=1), pooled

    def close(self) -> None:
        if self._redux is not None:
            for module in self._redux[1:]:
                try:
                    module.to("cpu")
                except Exception:
                    pass
            self._redux = None
        if self._flux is not None:
            try:
                self._flux.close()
            except Exception:
                pass
            self._flux = None
        if self._matter is not None:
            try:
                self._matter.unload()
            except Exception:
                pass
            self._matter = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def model_names(cls) -> list[str]:
        from pipeline.inpainting.inpainting_flux import DEV_MODEL_ID
        return [DEV_MODEL_ID, REDUX_MODEL_ID, "ZhengPeng7/BiRefNet"]

    # ── one part ────────────────────────────────────────────────────────────
    def _paint(self, prompt: str, seed: int, work: Path, tag: str,
               reference: Optional[PILImage.Image] = None) -> PILImage.Image:
        from util.image_utils import Image
        px = self.image_px
        base = PILImage.new("RGB", (px, px), (255, 255, 255))
        mask = PILImage.new("L", (px, px), 255)
        embeds = {}
        if reference is not None and self.reference_scale > 0.0:
            prompt_embeds, pooled = self._reference_embeds(prompt, reference, self.reference_scale)
            embeds = {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled}
        img = self._painter().inpaint(base, mask, work, prompt=prompt, num_inference_steps=self.steps,
                                      guidance_scale=self.guidance, strength=1.0, seed=seed, **embeds)
        img.save(work / f"{tag}_raw.png")
        seg = self._seg().segment(Image(img))
        alpha = np.asarray(seg.masks[0], dtype=bool)
        rgba = np.dstack([np.asarray(img.convert("RGB"), dtype=np.uint8), (alpha * 255).astype(np.uint8)])
        return PILImage.fromarray(rgba, "RGBA")

    @staticmethod
    def _cutout(rgba: PILImage.Image, pad: int = 4, trim_stem: bool = False) -> Optional[PILImage.Image]:
        """The opaque bounding box. With trim_stem, rows at the bottom whose opaque
        run is under 30% of the widest row are cut away first: FLUX paints a head
        on a length of stem however the prompt reads, and a card of the whole
        picture was two thirds stem (Rainier band20: heads "small coloured tips")."""
        a = np.asarray(rgba)[..., 3] > 127
        if a.sum() < 400:
            return None
        if trim_stem:
            widths = a.sum(axis=1)
            peak = int(widths.max())
            rows_on = np.where(widths > 0)[0]
            bottom = int(rows_on.max())
            while bottom > rows_on.min() and widths[bottom] < 0.3 * peak:
                bottom -= 1
            a[bottom + 1:] = False
            if a.sum() < 400:
                a = np.asarray(rgba)[..., 3] > 127
        rows, cols = np.where(a.any(axis=1))[0], np.where(a.any(axis=0))[0]
        y0, y1 = max(0, rows.min() - pad), min(a.shape[0], rows.max() + pad + 1)
        x0, x1 = max(0, cols.min() - pad), min(a.shape[1], cols.max() + pad + 1)
        return rgba.crop((x0, y0, x1, y1))

    @staticmethod
    def _drop_bark(rgba: PILImage.Image) -> PILImage.Image:
        """A hanging strand without the bough it was painted hanging from: FLUX hung
        the willow's branchlets on a black horizontal branch, and every curtain in
        the crown carried that bar (Paris, 2026-11-24). Dark pixels that are not
        green are made clear."""
        px = np.asarray(rgba.convert("RGBA")).copy()
        r, g, b = (px[..., i].astype(np.int16) for i in range(3))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        bark = (lum < 70) & (g - np.maximum(r, b) < 10)
        from scipy import ndimage
        bark = ndimage.binary_dilation(bark, iterations=2)  # and its soft outline
        px[..., 3] = np.where(bark, 0, px[..., 3])
        return PILImage.fromarray(px, "RGBA")

    # ── the atlas ───────────────────────────────────────────────────────────
    def build(self, entry: dict, species: dict, palette: dict, work: Path,
              photo: Optional[PILImage.Image] = None) -> Optional[dict]:
        """{"atlas": path, "parts": {part: {"uv": [u0, v0, u1, v1] (pixel rows top-down,
        0-1), "aspect": w/h}}} for the entry's form, or None when nothing could be made.
        With the photograph, a head the Species Director boxed is painted from it."""
        form = str(entry.get("form") or "spike")
        prompts = part_prompts(form, species)
        targets = {"bloom": (palette.get("bloom") or [None])[0], "foliage": (palette.get("foliage") or [None])[0]}
        head_crop = head_reference(photo, species) if self.reference_scale > 0.0 else None
        recipe = {
            "form": form, "prompts": prompts, "targets": targets, "px": self.image_px, "steps": self.steps,
            "guidance": self.guidance, "seed": self.seed, "v": 10,
        }
        if form == "weeping":
            recipe["strand"] = 2  # the bough is cut away from the strand
        if head_crop is not None:
            # Present only with a reference, so an unreferenced atlas keeps its cache.
            recipe["reference"] = [hashlib.sha256(head_crop.tobytes()).hexdigest()[:16], list(head_crop.size),
                                   self.reference_scale, self.reference_matte]
        key = hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()[:16]
        cached = self.cache_dir / f"atlas_{key}"
        if (cached / "atlas.png").exists() and (cached / "parts.json").exists():
            info = json.loads((cached / "parts.json").read_text())
            info["atlas"] = str(cached / "atlas.png")
            head = (info.get("parts") or {}).get("head")
            if head and not head.get("profile"):
                # An atlas painted before the head had a body: its outline is in the picture.
                u0, v0, u1, v1 = head["uv"]
                alpha = np.asarray(PILImage.open(cached / "atlas.png").convert("RGBA"))[..., 3]
                H, W = alpha.shape
                head["profile"] = outline_profile(alpha[int(v0 * H):int(v1 * H), int(u0 * W):int(u1 * W)])
            return info
        work = Path(work) / f"atlas_{key}"
        work.mkdir(parents=True, exist_ok=True)
        atlas = PILImage.new("RGBA", (ATLAS_PX, ATLAS_PX), (0, 0, 0, 0))
        parts: dict[str, dict] = {}
        for i, (part, pal, prompt) in enumerate(prompts[: len(_CELLS)]):
            seed = self.seed + i * 31 + sum(ord(c) for c in part)
            try:
                reference = None
                if part == "head" and head_crop is not None:
                    reference = self.reference_image(head_crop, work, f"{entry.get('id', 'species')}_{part}")
                rgba = self._paint(prompt, seed, work, f"{entry.get('id', 'species')}_{part}", reference=reference)
            except Exception as exc:
                self.log.warning(f"  atlas: {part} of {species.get('name')} failed: {exc}")
                continue
            if part == "strand":
                rgba = self._drop_bark(rgba)
            cut = self._cutout(rgba, trim_stem=(part == "head"))
            if cut is None:
                self.log.warning(f"  atlas: {part} of {species.get('name')}: the matte found nothing")
                continue
            if part == "strand":
                cut = cut.transpose(PILImage.FLIP_TOP_BOTTOM)
            cut = PILImage.fromarray(decontaminate(np.asarray(cut)), "RGBA")
            target = targets.get(pal)
            if target is not None:
                # Foliage was measured delit and in the meadow's shade; its hue is
                # the photograph's, its darkness is not what a lit leaf shows.
                # A conifer's spray takes most of the measured darkness: a subalpine
                # fir is near black in the photograph, and the painted spray lit by the
                # scene's sun read as "bright yellow-green pillars" (band22).
                # A leaf takes nearly half the measured darkness: at a quarter the
                # painted leaves stayed FLUX's bright yellow-green (band26).
                lightness = 0.5 if pal == "bloom" else (0.8 if part in ("spray", "strand") else 0.45)
                chroma = 0.8 if pal == "bloom" else (1.0 if part in ("spray", "strand") else 0.85)
                # A bloom keeps 40% of the painting's own colour: the plan's colour
                # words made the lousewort's cream spike a pure yellow (band24,
                # "cartoon colors"), and FLUX knows what the named species looks like.
                strength = 0.6 if pal == "bloom" else 0.85
                cut = PILImage.fromarray(recolour(np.asarray(cut), target, strength=strength, lightness=lightness,
                                                  chroma=chroma), "RGBA")
            cut = PILImage.fromarray(alpha_bleed(np.asarray(cut)), "RGBA")
            w, h = cut.size
            scale = min((CELL_PX - 8) / w, (CELL_PX - 8) / h)
            cut = cut.resize((max(1, int(w * scale)), max(1, int(h * scale))), PILImage.LANCZOS)
            cx, cy = _CELLS[i]
            x0, y0 = cx + (CELL_PX - cut.width) // 2, cy + (CELL_PX - cut.height) // 2
            atlas.paste(cut, (x0, y0))
            parts[part] = {
                "uv": [x0 / ATLAS_PX, y0 / ATLAS_PX, (x0 + cut.width) / ATLAS_PX, (y0 + cut.height) / ATLAS_PX],
                "aspect": round(cut.width / max(cut.height, 1), 4), "target": pal,
            }
            if part == "head":
                parts[part]["profile"] = outline_profile(np.asarray(cut)[..., 3])
            cut.save(work / f"{entry.get('id', 'species')}_{part}_cut.png")
        if not parts:
            return None
        cached.mkdir(parents=True, exist_ok=True)
        atlas = PILImage.fromarray(alpha_bleed(np.asarray(atlas)), "RGBA")
        atlas.save(cached / "atlas.png")
        info = {"parts": parts, "palette_origin": list(PALETTE_ORIGIN), "palette_px": PALETTE_PX, "size": ATLAS_PX}
        (cached / "parts.json").write_text(json.dumps(info, indent=1))
        info["atlas"] = str(cached / "atlas.png")
        return info
