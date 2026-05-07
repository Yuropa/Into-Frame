from path_utils import add_project_paths, checkpoints_path
add_project_paths()

from pathlib import Path
from typing import Any
from PIL import Image
import numpy as np
from pipeline.model_generation.model_generation_base_imp import ModelGeneratorBase
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

sam3d_path = Path(sam3d_objects.__file__).parent.parent
sys.path.insert(0, str(sam3d_path / "notebook"))

from inference import Inference

class ModelGenerator(ModelGeneratorBase):
    def setup(self):
        self.image_resolution = 1024
        config_file = checkpoints_path() / "hf" / "pipeline.yaml"
        self.model = Inference(config_file=config_file, compile=False)

    def meshify(self, temp_path: Path, input: Image, mask: np.ndarray = None) -> Any:
        image_np = np.array(input.convert("RGB")).astype(np.uint8)

        # SAM3D requires a mask — use full image mask if none provided
        if mask is None:
            mask = np.ones(image_np.shape[:2], dtype=bool)

        output = self.model(image=image_np, mask=mask)

        # Export the gaussian splat as a .ply
        if "gaussian" in output:
            gaussian = output["gaussian"][0]
            ply_path = str(temp_path / "output.ply")
            gaussian.save_ply(ply_path)  # adjust if their API uses a different method

        return output


if __name__ == "__main__":
    ModelGenerator.run()