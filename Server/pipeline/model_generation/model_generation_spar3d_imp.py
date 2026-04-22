import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent / ".." / ".." / ".." / "lib" / "packages"))

import torch
import socket
import json
import base64
from io import BytesIO
from PIL import Image
from transparent_background import Remover
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, remove_background

class ModelGenerator():
    def __init__(self, device, temp_path) -> None:
        self.device = device
        self.temp_path = temp_path
        self.image_resolution = 1024
        self.foreground_ratio = 1.3
        self.model = SPAR3D.from_pretrained(
            "stabilityai/stable-point-aware-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=False
        )
        self.model.to(device)
        self.model.eval()
        self.bg_remove = Remover(device=device)

    def meshify(self, image_b64: str):
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        
        image.save(str(self.temp_path / "output.png"))

        cleaned_image = remove_background(image, self.bg_remove)
        cleaned_image = foreground_crop(cleaned_image, self.foreground_ratio)

        cleaned_image.save(str(self.temp_path / "cleaned_image.png"))

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
        with open(str(self.temp_path / "debug.json"), "w") as f:
            json.dump(debug, f, indent=2)

        mesh.export(str(self.temp_path / "mesh.glb"), include_normals=True)
        if "point_clouds" in glob_dict:
            glob_dict["point_clouds"][0].export(str(self.temp_path / "points.ply"))
        # Serialize mesh and return as JSON
        return {"vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"    
    sock_path = sys.argv[2]

    temp_path = Path(sys.argv[3])
    temp_path.mkdir(parents=True, exist_ok=True) 

    sys.stdout = open(str(temp_path / "stdout.log"), "w", buffering=1)
    sys.stderr = open(str(temp_path / "stderr.log"), "w", buffering=1)

    generator = ModelGenerator(device, temp_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    json_in = sock.makefile('r')
    json_out = sock.makefile('w')

    print(json.dumps({"status": "ready"}), file=json_out, flush=True)

    for line in json_in:
        request = json.loads(line.strip())
        if request["action"] == "meshify":
            result = generator.meshify(request["image_b64"])
            print(json.dumps(result), file=json_out, flush=True)
        elif request["action"] == "exit":
            break

    sock.close()