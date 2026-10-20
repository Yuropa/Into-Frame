import torch
from pathlib import Path
from remote_connection.remote_client import RemoteClient
from util.intrinsic_utils import IntrinsicImages
from util.image_utils import Image

class ImageIntrinsics(RemoteClient):
    """
    Bridges to IntrinsicDiffusion (Luo et al., SIGGRAPH 2024), running in its own
    conda env, to predict per-pixel albedo (unlit reflectance) and surface normal
    maps from a single image.
    """

    # "marigold": Marigold-IID lighting (prs-eth/marigold-iid-lighting-v1-1) in the
    # frame env -- its albedo takes the light's colour out, not only its intensity,
    # which is what a sunset capture needs. "intrinsicdiffusion": the original
    # model in its own env.
    BACKENDS = {
        "marigold": ("frame", "image_intrinsics_marigold_imp.py"),
        "intrinsicdiffusion": ("intrinsicdiffusion", "image_intrinsics_imp.py"),
    }

    def __init__(self, device: torch.device, backend: str = "intrinsicdiffusion") -> None:
        backend = str(backend or "intrinsicdiffusion").lower()
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown intrinsic backend {backend!r}; one of {sorted(self.BACKENDS)}")
        conda_env, script = self.BACKENDS[backend]
        self.backend = backend
        script_path = Path(__file__).parent / script

        super().__init__(
            device=device,
            conda_env=conda_env,
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["ptx0/pseudo-journey-v2", "prs-eth/marigold-iid-lighting-v1-1"]

    def intrinsic_images(
        self,
        input: Image,
        temp_path: Path,
        resolution: int = 1024,
        agg_num: int = 4,
        on_progress=None,
    ) -> IntrinsicImages:
        result = self.send(
            action="intrinsics",
            input={"image": input.rgb(), "resolution": resolution, "agg_num": agg_num},
            temp_path=temp_path,
            on_progress=on_progress,
        )
        return IntrinsicImages(result)
