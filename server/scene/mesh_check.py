"""
Does a reconstructed mesh still look like its class?

The geometric guards (aspect against the crop, axis-aligned faces) catch some
failed reconstructions, but not the shapes a reconstructor falls back to for a
tree it cannot read: Rainier's tree_1 a brown cigar, tree_2 a grey pillar, and
Tree-D's tree_3 a flat yellow sheet, all placed in the scene as black slabs.
This renders the mesh from two sides on white and asks CLIP whether that is
"a {class}" or a blob, rock, pillar, sheet, box, sack or log. Measured: those
three trees 0.13-0.26, a good conifer 0.97, flowers 0.99, Paris's boats 0.99,
tower 0.998, dock 0.81, a flat tree card 0.46.
"""
from __future__ import annotations

import math
import os

import numpy as np

_NEGATIVES = (
    "a shapeless blob", "a rock", "a stone pillar", "a flat sheet of paper",
    "a cardboard box", "a sack", "a log",
)
_MODEL = "openai/clip-vit-base-patch32"
_clip = None


def _load():
    global _clip
    if _clip is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip = (CLIPProcessor.from_pretrained(_MODEL), CLIPModel.from_pretrained(_MODEL).eval())
    return _clip


def _views(tm, size: int = 224) -> list:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not hasattr(np, "infty"):
        np.infty = np.inf
    import pyrender
    from PIL import Image as PILImage
    m = tm.copy()
    m.apply_translation(-m.bounds.mean(axis=0))
    ext = float(np.max(m.extents)) or 1.0
    out = []
    for yaw in (0.0, 90.0):
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.8, 0.8, 0.8])
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        a = math.radians(yaw)
        d = ext * 1.9
        eye = np.array([d * math.sin(a), ext * 0.15, d * math.cos(a)])
        f = -eye / np.linalg.norm(eye)
        r = np.cross(f, [0.0, 1.0, 0.0]); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        pose = np.eye(4); pose[:3, 0] = r; pose[:3, 1] = u; pose[:3, 2] = -f; pose[:3, 3] = eye
        sc.add(pyrender.PerspectiveCamera(yfov=math.radians(35)), pose=pose)
        sc.add(pyrender.DirectionalLight(intensity=2.5), pose=pose)
        renderer = pyrender.OffscreenRenderer(size, size)
        try:
            colour, _ = renderer.render(sc)
        finally:
            renderer.delete()
        out.append(PILImage.fromarray(colour))
    return out


def class_likeness(tm, cls: str) -> float:
    """Probability mass CLIP puts on "a {cls}" over the failure shapes, averaged
    over two views. Raises if rendering is unavailable (callers treat that as
    no evidence)."""
    import torch
    proc, model = _load()
    name = str(cls).replace("_", " ")
    positives = [f"a photo of a {name}", f"a 3D model of a {name}"]
    images = _views(tm)
    with torch.no_grad():
        inputs = proc(text=positives + list(_NEGATIVES), images=images, return_tensors="pt", padding=True)
        probs = model(**inputs).logits_per_image.softmax(-1).numpy()
    return float(probs[:, :len(positives)].sum(axis=1).mean())
