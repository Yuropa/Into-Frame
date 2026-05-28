import torch
from pathlib import Path

from pipeline.model_generation.model_generation_base import ModelGeneratorBase


class ModelGeneratorSam3D(ModelGeneratorBase):
    """
    Mesh generator backed by SAM 3 (Segment Anything with Concepts).

    SAM 3 is a segmentation model, not a 3D reconstructor. This generator uses
    SAM 3's text-prompted segmentation to get a clean object mask, then extrudes
    the 2D silhouette into a shallow 3D mesh and applies the masked image as a
    texture. Results are cookie-cutter style — accurate silhouette, limited depth.

    Note: trimesh, shapely, and opencv-python must be present in the sam3d env.
    Add them to the setup script if the subprocess fails to import them:
        conda run -n sam3d pip install trimesh shapely opencv-python
    """

    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "model_generation_sam3d_imp.py"
        super().__init__(
            device=device,
            conda_env="sam3d",
            script_path=script_path,
        )

    @classmethod
    def model_names(cls) -> list[str]:
        # Downloaded via setup.sh from facebook/sam-3d-objects — not a HF snapshot
        return []
