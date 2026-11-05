import os
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))

from path_utils import add_project_paths, lib_path, checkpoints_path, add_system_path
add_project_paths()

add_system_path(lib_path() / "SAM3D")

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from PIL import Image as PILImage

from pipeline.model_generation.model_generation_base_imp import ModelGeneratorBase


class ModelGenerator(ModelGeneratorBase):
    """
    Uses SAM3D (sam-3d-objects) to reconstruct a 3D mesh from a single image.

    The pipeline takes an RGBA image (alpha encodes the object mask) and produces
    a textured trimesh via sparse-structure + structured-latent diffusion.
    """

    def setup(self):
        config_path = checkpoints_path() / "hf" / "pipeline.yaml"
        config = OmegaConf.load(str(config_path))
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = str(config_path.parent)
        print(f"Loading SAM3D pipeline from {config_path}")
        self._pipeline = instantiate(config)
        print("SAM3D pipeline loaded")

    def meshify(self, temp_path: Path, input: PILImage.Image, seed: int = 0) -> Any:
        image_np = np.array(input.convert("RGBA"))

        if input.mode == "RGBA":
            mask = (image_np[:, :, 3] > 0).astype(np.uint8) * 255
        else:
            mask = np.full((image_np.shape[0], image_np.shape[1]), 255, dtype=np.uint8)

        rgba = np.concatenate([image_np[:, :, :3], mask[:, :, None]], axis=-1)

        try:
            output = self._pipeline.run(
                rgba,
                mask=None,
                seed=seed,
                with_mesh_postprocess=True,
                with_texture_baking=True,
                use_vertex_color=False,
            )
        except torch.cuda.OutOfMemoryError:
            # Texture baking (bake_texture's "opt" mode) holds every rasterized
            # view on GPU for the whole optimization -- the single heaviest step
            # in this pipeline. Retry once vertex-color-only (no baking) so this
            # object still comes back with a mesh instead of losing it entirely.
            print("SAM3D texture baking OOM'd -- retrying without texture baking", flush=True)
            torch.cuda.empty_cache()
            output = self._pipeline.run(
                rgba,
                mask=None,
                seed=seed,
                with_mesh_postprocess=True,
                with_texture_baking=False,
                use_vertex_color=True,
            )
        self._write_pose(temp_path, output)
        return output["glb"]

    @staticmethod
    def _write_pose(temp_path: Path, output: dict) -> None:
        """Record where the photographer stood relative to the mesh.

        SAM3D reconstructs the object in a canonical, gravity-aligned frame and
        predicts the pose (quaternion, translation, scale) that carries it into the
        input camera's PyTorch3D frame (+X left, +Y up, +Z forward): p_cam =
        (s * p_local) @ R + t. The GLB keeps the canonical frame re-axed Y-up
        (to_glb: (x, y, z) -> (x, z, -y)), so without the pose nothing downstream
        knows which side of the mesh the photograph shows. The direction from the
        object to the camera is written in the GLB's own frame alongside the raw
        pose; the generator client reads it back (model_generation_base).
        """
        try:
            from pytorch3d.transforms import quaternion_to_matrix
            import json
            q = output["rotation"].reshape(-1, 4)[0].detach().float().cpu()
            t = output["translation"].reshape(-1, 3)[0].detach().float().cpu().numpy()
            s = output["scale"].reshape(-1)[0:3].detach().float().cpu().numpy()
            R = quaternion_to_matrix(q[None])[0].numpy()
            norm = float(np.linalg.norm(t))
            # A decoder that leaves the object at the origin says nothing about
            # the view direction; the camera then looks straight down +Z at it.
            d_cam = (-t / norm) if norm > 1e-6 else np.array([0.0, 0.0, -1.0])
            d_local = d_cam @ R.T
            d_glb = np.array([d_local[0], d_local[2], -d_local[1]], dtype=float)
            with open(temp_path / "pose.json", "w") as f:
                json.dump({
                    "rotation_wxyz": [float(v) for v in q],
                    "translation": [float(v) for v in t],
                    "scale": [float(v) for v in s],
                    "toward_camera_glb": [float(v) for v in d_glb],
                }, f, indent=2)
        except Exception as e:  # the mesh is still good without it
            print(f"SAM3D pose not recorded: {e}", flush=True)


if __name__ == "__main__":
    ModelGenerator.run()
