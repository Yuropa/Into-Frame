from path_utils import add_project_paths, lib_path, checkpoints_path, add_system_path
add_project_paths()

add_system_path(lib_path() / "SAM3D")

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image as PILImage
from shapely.geometry import Polygon

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from pipeline.model_generation.model_generation_base_imp import ModelGeneratorBase


# Text prompt used to locate the main subject in each crop.
_SEGMENT_PROMPT = "object"

# How thick (in normalised model units) to extrude the silhouette.
_EXTRUSION_DEPTH = 0.12


class ModelGenerator(ModelGeneratorBase):
    """
    Uses SAM 3 image segmentation to get a precise object mask, then builds a
    shallow extruded-silhouette mesh with the masked image as a texture.
    """

    def setup(self):
        checkpoint_dir = checkpoints_path() / "hf"
        checkpoints = sorted(
            list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No SAM3 checkpoint (.pt / .pth) found in {checkpoint_dir}"
            )
        checkpoint_path = checkpoints[0]
        print(f"Loading SAM3 from {checkpoint_path}")

        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
            device=str(self.device),
        )
        self.processor = Sam3Processor(model=model, device=str(self.device))
        print("SAM3 loaded")

    def meshify(self, temp_path: Path, input: PILImage.Image, seed: int = 0) -> Any:
        image = input.convert("RGB")
        w, h = image.size
        aspect = h / w

        # --- Segmentation --------------------------------------------------
        state = self.processor.set_image(image)
        state = self.processor.set_text_prompt(prompt=_SEGMENT_PROMPT, state=state)

        masks = state.get("masks")    # (N, H_feat, W_feat) bool tensor or None
        scores = state.get("scores")  # (N,) float tensor or None

        if masks is not None and len(masks) > 0:
            best = int(scores.argmax()) if scores is not None else 0
            raw_mask = masks[best].cpu().numpy().astype(np.uint8) * 255
            # Resize mask to original image dimensions
            mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 127
        else:
            # No detection — treat the whole crop as the object
            mask = np.ones((h, w), dtype=bool)

        # --- Texture: image with background removed (RGBA) -----------------
        img_array = np.array(image.convert("RGBA"))
        img_array[:, :, 3] = (mask * 255).astype(np.uint8)
        texture = PILImage.fromarray(img_array, "RGBA")
        texture.save(str(temp_path / "texture.png"))

        # --- Mesh: extrude silhouette contour ------------------------------
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            return self._extruded_mesh(contours, w, h, aspect, texture)
        return self._quad_mesh(aspect, texture)

    # ------------------------------------------------------------------
    def _extruded_mesh(self, contours, w, h, aspect, texture) -> trimesh.Trimesh:
        """Extrude the largest mask contour into a shallow 3D mesh."""
        largest = max(contours, key=cv2.contourArea)
        eps = 0.015 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2).astype(float)

        if len(approx) < 3:
            return self._quad_mesh(aspect, texture)

        # Normalise to centred [-0.5, 0.5] × [-aspect/2, aspect/2]
        pts = np.column_stack([
            approx[:, 0] / w - 0.5,
            aspect * (0.5 - approx[:, 1] / h),
        ])

        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_valid or poly.area < 1e-4:
                return self._quad_mesh(aspect, texture)

            depth = min(_EXTRUSION_DEPTH, max(0.04, poly.area ** 0.5 * 0.25))
            mesh = trimesh.creation.extrude_polygon(poly, depth)
            # Centre along Z so the object sits at origin
            mesh.apply_translation([0.0, 0.0, -depth / 2])
            # Apply a simple double-sided material
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.full((len(mesh.vertices), 4), [200, 200, 200, 255], dtype=np.uint8),
            )
            return mesh
        except Exception as e:
            print(f"Extrusion failed ({e}), falling back to quad")
            return self._quad_mesh(aspect, texture)

    def _quad_mesh(self, aspect, texture) -> trimesh.Trimesh:
        """Textured flat quad covering the full crop, as a fallback."""
        h = aspect
        vertices = np.array([
            [-0.5, -h / 2, 0],
            [ 0.5, -h / 2, 0],
            [ 0.5,  h / 2, 0],
            [-0.5,  h / 2, 0],
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture, doubleSided=True
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        return mesh


if __name__ == "__main__":
    ModelGenerator.run()
