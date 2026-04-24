from pathlib import Path
from typing import Any

import torch
import base64
from io import BytesIO
from PIL import Image
from util.json_utils import write_json
from abc import ABC, abstractmethod
from remote_connection.remote_server import RemoteServer

class ModelGeneratorBase(RemoteServer, ABC):
    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "meshify":
            return self._run_meshify(temp_path, input)
        raise ValueError(f"Unknown action: {action}")

    
    @abstractmethod
    def meshify(self, temp_path: Path, input: Image) -> Any:
        pass
    
    def _run_meshify(self, temp_path: Path, input: Any) -> Any:
        image_data = base64.b64decode(input)
        image = Image.open(BytesIO(image_data))
        
        image.save(str(temp_path / "output.png"))

        with torch.no_grad():
            mesh = self.meshify(
                temp_path=temp_path,
                input=image
            )

        debug = {
            "mesh_type": str(type(mesh)),
            "mesh_repr": str(mesh),
            "has_vertices": hasattr(mesh, "vertices"),
            "has_faces": hasattr(mesh, "faces"),
            "vertex_count": len(mesh.vertices) if hasattr(mesh, "vertices") else None,
            "face_count": len(mesh.faces) if hasattr(mesh, "faces") else None,
        }
        with open(str(temp_path / "debug.json"), "w") as f:
            write_json(debug, f)

        mesh_path = temp_path / "mesh.glb"
        mesh.export(str(mesh_path), include_normals=True)

        # Serialize mesh and return as JSON
        return str(mesh_path)