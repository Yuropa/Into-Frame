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

        output = self._pipeline.run(
            rgba,
            mask=None,
            seed=seed,
            with_mesh_postprocess=True,
            with_texture_baking=False,
            use_vertex_color=True,
        )
        return output["glb"]


if __name__ == "__main__":
    ModelGenerator.run()
