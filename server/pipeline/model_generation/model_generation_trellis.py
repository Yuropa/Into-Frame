import logging
import numpy as np
import torch
import trimesh
from pathlib import Path

from scene.mesh import Mesh
from util.image_utils import Image

from pipeline.model_generation.model_generation_base import ModelGeneratorBase

_log = logging.getLogger(__name__)


# Scene Generation's sheet gate (_MIN_MESH_THICKNESS_FRACTION there): thinnest extent
# over the second largest. A candidate under it is disqualified here, whatever its
# other terms, because every instance of it would be billboarded downstream. The
# 05:25 Rainier run kept a 0.25-thick lupine draw at score 0.5 and then placed none.
MIN_THICKNESS_FRACTION = 0.3


class ImplausibleReconstruction(RuntimeError):
    """Every candidate draw disagreed with the picture; the crop stays billboard-only."""


class ModelGeneratorTrellis(ModelGeneratorBase):
    # TRELLIS.2 is trained on PBR renders and its base-colour map is an albedo
    # estimate, so it wants the photographed (lit) crop: shading is its main depth
    # cue. Fed a delit crop it reads the subject as a flat picture and returns a
    # relief (the 2026-10-14 Rainier run: every flower bucket came back 0.06-0.3
    # thick and was rejected as a sheet).
    produces_albedo = True

    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "model_generation_trellis_imp.py"

        super().__init__(
            device=device, 
            conda_env="trellis2", 
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["microsoft/TRELLIS.2-4B"]

    # Reconstructions to draw per crop; the one whose shape agrees best with the
    # crop is kept (see _plausibility). TRELLIS.2 is fast (15-30 s a draw) but
    # erratic on cutouts of vegetation: the same lupine crop came back as an
    # upright plant, a solid block and a flat bed lying on the ground across
    # seeds and small input changes, and only the first survives Scene
    # Generation's shape gates. Drawing a few and scoring them is cheaper than
    # a wasted mesh.
    candidates = 4
    # Best candidate score (lower is better) still worth shipping as a mesh. On
    # the Rainier and Paris draws the upright plants, the boat, the person and
    # the tree scored 0.1-0.5, the bridge 0.5; a solid block with the picture
    # painted on it 1.2 (Scene Generation's gates cannot tell a block from a
    # bush, so it has to be stopped here), a stick for a boat 1.2; sheets, beds
    # and strips 2.0-5.0. Above this the crop stays billboard-only, by way of the
    # exception Asset Generation already catches per mesh.
    max_score = 0.8

    def meshify(self, image: Image, temp_path: Path, seed: int = 0) -> Mesh:
        best: "tuple[float, Mesh, str] | None" = None
        failures: list[str] = []
        for k in range(max(1, int(self.candidates))):
            try:
                mesh = super().meshify(image, temp_path, seed=seed + k)
            except Exception as e:
                # One draw can die inside TRELLIS (CuMesh fill_holes raised a CUDA
                # "invalid argument", an empty shape lattice hit a max() over no
                # elements); the other seeds are still worth drawing.
                failures.append(f"seed {seed + k}: {str(e).splitlines()[0][:120]}")
                _log.info(f"    candidate seed {seed + k} failed: {failures[-1]}")
                continue
            # o_voxel's to_glb already hands back a Y-up, image-aligned mesh
            # (image x -> +x, image up -> +y, depth -> z), the frame the scene
            # uses. No axis conversion: the earlier -90 deg about X was a
            # misreading of a bounding box inflated by floating debris along z,
            # and it laid every mesh flat.
            self._drop_debris(mesh.mesh)
            score, why = self._plausibility(mesh.mesh, image)
            _log.info(f"    candidate seed {seed + k}: {why} -> score {score:.2f}")
            if best is None or score < best[0]:
                best = (score, mesh, why)
        if best is None:
            raise ImplausibleReconstruction("every candidate draw failed: " + "; ".join(failures))
        if best[0] > self.max_score:
            raise ImplausibleReconstruction(
                f"no candidate agreed with the picture (best score {best[0]:.2f} > {self.max_score}: {best[2]})"
            )
        if self.candidates > 1:
            _log.info(f"    kept the candidate with score {best[0]:.2f} ({best[2]})")
        return best[1]

    @staticmethod
    def _plausibility(tm: trimesh.Trimesh, image: Image, grid: int = 32) -> "tuple[float, str]":
        """How well the mesh's shape agrees with the picture it was made from;
        lower is better. Three terms, each one a way a draw has gone wrong:

        aspect    the mesh's front-view box (x over y) against the cutout's
                  pixel box -- a tall stalk that came back as a wide bed
                  lying on the ground scores badly here;
        fill      the fraction of that box the silhouette covers, mesh against
                  alpha -- a solid block with the picture painted on it fills
                  its box, a plant does not;
        thickness Scene Generation's own sheet test (thinnest extent over the
                  second largest, under 0.4 is carded), so a relief is only
                  chosen when nothing better came back.
        """
        pil = image.image.convert("RGBA")
        alpha = np.asarray(pil)[..., 3] > 8
        ys, xs = np.nonzero(alpha)
        if len(xs) == 0 or len(tm.vertices) == 0:
            return 0.0, "no alpha"
        crop_aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        sub = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        # Occupancy on the same coarse grid for both silhouettes.
        gy = (np.arange(sub.shape[0]) * grid // sub.shape[0])
        gx = (np.arange(sub.shape[1]) * grid // sub.shape[1])
        crop_cells = np.zeros((grid, grid), dtype=bool)
        yy, xx = np.nonzero(sub)
        crop_cells[gy[yy], gx[xx]] = True
        crop_fill = float(crop_cells.mean())

        v = tm.vertices
        ext = v.max(0) - v.min(0)
        mesh_aspect = float(ext[0] / max(ext[1], 1e-6))
        q = ((v[:, :2] - v[:, :2].min(0)) / np.maximum(ext[:2], 1e-6) * (grid - 1)).astype(int)
        mesh_cells = np.zeros((grid, grid), dtype=bool)
        mesh_cells[q[:, 1], q[:, 0]] = True
        mesh_fill = float(mesh_cells.mean())

        e = np.sort(ext)
        thickness = float(e[0] / max(e[1], 1e-6))
        # An elongated subject may have been photographed end-on (a boat bow-on),
        # so the picture is allowed to match either the broadside (x/y) or the
        # end-on (z/y) silhouette, as Scene Generation's own aspect gate does.
        endon_aspect = float(ext[2] / max(ext[1], 1e-6))
        aspect_term = min(
            abs(np.log(max(mesh_aspect, 1e-3) / max(crop_aspect, 1e-3))),
            abs(np.log(max(endon_aspect, 1e-3) / max(crop_aspect, 1e-3))),
        )
        # A silhouette sparser than the picture is normal (vertices thin out on
        # a plant); one much FULLER than the picture is a block with the picture
        # painted on it, which is the failure nothing downstream can catch.
        fill_term = 3.0 * max(0.0, mesh_fill - crop_fill) + 1.0 * max(0.0, crop_fill - mesh_fill)
        thick_term = 3.0 * max(0.0, 0.4 - thickness)
        if thickness < MIN_THICKNESS_FRACTION:
            # Scene Generation would card every instance of this mesh anyway, so
            # it must not win over a draw it would place.
            thick_term += 10.0
        score = aspect_term + fill_term + thick_term
        why = (f"extents {np.round(ext, 3)}, aspect {mesh_aspect:.2f} vs crop {crop_aspect:.2f}, "
               f"fill {mesh_fill:.2f} vs crop {crop_fill:.2f}, thickness {thickness:.2f}")
        return float(score), why

    @staticmethod
    def _drop_debris(tm: trimesh.Trimesh, min_face_fraction: float = 0.01, margin: float = 0.35,
                     percentile: float = 5.0) -> None:
        """Remove small disconnected fragments floating away from the subject.

        TRELLIS leaves specks of geometry well outside the object (on a lupine
        test the plant was 0.27 x 0.26 x 0.25 and the specks stretched its box to
        0.96 along z). They are invisible but they set the bounding box, so
        fit_to_box shrinks the plant to a fraction of its target size and the
        shape gates in Scene Generation measure the specks, not the subject.

        The subject's box is taken from the vertex distribution (the
        `percentile`..100-`percentile` range on each axis), not from the largest
        connected component: the remeshed output is a soup of thousands of small
        pieces (16,166 components, the largest 360 faces, on that lupine) and the
        largest of them is a small part of the plant. The specks sit on the far
        face of the voxel box (z = +0.5) and amount to 2-4% of the vertices, so a
        5% trim on each side excludes them while the box, grown by `margin`,
        still reaches the plant's own tips. A
        component is debris when it holds under `min_face_fraction` of the faces
        AND its centroid lies outside that box grown by `margin` of its longest
        side.
        """
        if len(tm.faces) == 0:
            return
        labels = trimesh.graph.connected_component_labels(tm.face_adjacency, node_count=len(tm.faces))
        counts = np.bincount(labels, minlength=1)
        if len(counts) < 2:
            return
        lo = np.percentile(tm.vertices, percentile, axis=0)
        hi = np.percentile(tm.vertices, 100.0 - percentile, axis=0)
        pad = margin * float((hi - lo).max())
        keep = np.ones(len(tm.faces), dtype=bool)
        small = np.flatnonzero(counts < min_face_fraction * len(tm.faces))
        dropped = 0
        if len(small):
            # Centroid per component in one pass.
            face_centres = tm.vertices[tm.faces].mean(1)
            sums = np.zeros((len(counts), 3)); np.add.at(sums, labels, face_centres)
            centres = sums / np.maximum(counts, 1)[:, None]
            outside = np.any(centres < lo - pad, axis=1) | np.any(centres > hi + pad, axis=1)
            debris = small[outside[small]]
            if len(debris):
                keep = ~np.isin(labels, debris)
                dropped = len(debris)
        if dropped:
            before = tm.extents.copy()
            tm.update_faces(keep)
            tm.remove_unreferenced_vertices()
            _log.info(
                f"    dropped {dropped} floating fragment(s) ({int((~keep).sum())} faces); "
                f"extents {np.round(before, 3)} -> {np.round(tm.extents, 3)}"
            )
