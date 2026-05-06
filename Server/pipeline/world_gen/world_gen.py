import torch
from pathlib import Path
import numpy as np
from io import BytesIO
from PIL import Image
from remote_connection.remote_client import RemoteClient 
from pipeline.model_generation.model_generation_base import ModelGeneratorBase

class WorldGen(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "world_gen_imp.py"

        super().__init__(
            device=device, 
            conda_env="worldgen", 
            script_path=script_path,
            env_options={
                "XFORMERS_DISABLED": 1
            }
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return []

    def generate(self, input: Image, temp_path: Path) -> Image:
        return self.send(action="generate", input=input, temp_path=temp_path)