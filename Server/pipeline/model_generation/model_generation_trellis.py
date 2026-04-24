import torch
from pathlib import Path

from pipeline.model_generation.model_generation_base import ModelGeneratorBase

class ModelGeneratorTrellis(ModelGeneratorBase):
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