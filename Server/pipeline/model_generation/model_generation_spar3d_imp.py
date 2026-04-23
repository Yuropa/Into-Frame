import sys
from pathlib import Path
from typing import Any
sys.path.append(str(Path(__file__).parent / ".." / ".." / ".." / "lib" / "packages"))
sys.path.append(str(Path(__file__).parent / ".." / ".."))

import torch
import base64
from io import BytesIO
from PIL import Image
from transparent_background import Remover
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, remove_background
from util.json_utils import write_json
from remote_connection.remote_server import RemoteServer

class ModelGenerator(RemoteServer):
    def __init__(self) -> None:
        super().__init__()

        self.image_resolution = 1024
        self.foreground_ratio = 1.3
        self.model = SPAR3D.from_pretrained(
            "stabilityai/stable-point-aware-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=False
        )
        self.model.to(self.device)
        self.model.eval()
        self.bg_remove = Remover(device=self.device)

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "meshify":
            return self._meshify(temp_path, input)
        raise ValueError(f"Unknown action: {action}")
    
    def _meshify(self, temp_path: Path, input: Any) -> Any:
        image_data = base64.b64decode(input)
        image = Image.open(BytesIO(image_data))
        
        image.save(str(temp_path / "output.png"))

        cleaned_image = remove_background(image, self.bg_remove)
        cleaned_image = foreground_crop(cleaned_image, self.foreground_ratio)

        cleaned_image.save(str(temp_path / "cleaned_image.png"))

        with torch.no_grad():
            mesh, glob_dict = self.model.run_image(
                cleaned_image,
                bake_resolution=self.image_resolution,
                remesh="quad",
                vertex_count=-1,
                return_points=True
            )

        debug = {
            "mesh_type": str(type(mesh)),
            "mesh_repr": str(mesh),
            "glob_dict_keys": list(glob_dict.keys()),
            "has_vertices": hasattr(mesh, "vertices"),
            "has_faces": hasattr(mesh, "faces"),
            "vertex_count": len(mesh.vertices) if hasattr(mesh, "vertices") else None,
            "face_count": len(mesh.faces) if hasattr(mesh, "faces") else None,
        }
        with open(str(temp_path / "debug.json"), "w") as f:
            write_json(debug, f)

        mesh_path = temp_path / "mesh.glb"
        mesh.export(str(mesh_path), include_normals=True)
        if "point_clouds" in glob_dict:
            glob_dict["point_clouds"][0].export(str(temp_path / "points.ply"))
        # Serialize mesh and return as JSON
        return str(mesh_path)

if __name__ == "__main__":
    ModelGenerator.run()