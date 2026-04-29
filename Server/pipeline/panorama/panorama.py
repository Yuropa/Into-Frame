import torch
from pathlib import Path
import numpy as np
from io import BytesIO
from PIL import Image
from remote_connection.remote_client import RemoteClient 
from pipeline.model_generation.model_generation_base import ModelGeneratorBase

class DepthResult:
    def __init__(self, result) -> None:
        self.depth = result['depth']
        self.confidence = result['conf']
        self.extrinsics = result['extrinsics']
        self.intrinsics = result['intrinsics']

class ImageDepth(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "depth_imp.py"

        super().__init__(
            device=device, 
            conda_env="depthanything", 
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["depth-anything/da3nested-giant-large"]

    def depth(self, input: Image, temp_path: Path) -> DepthResult:
        result = self.send(action="depth", input=self.encode_image(input), temp_path=temp_path)
        return DepthResult(result)