import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent / ".." / ".." / ".." / "lib" / "packages"))
sys.path.append(str(Path(__file__).parent / ".." / ".."))

import torch
import socket
import base64
from io import BytesIO
from PIL import Image
from transparent_background import Remover
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, remove_background
from util.json_utils import parse_json, write_json
from util.device_utils import clean_device_cache

class ModelGenerator():
    def __init__(self, device) -> None:
        self.device = device
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

    def meshify(self, image_b64: str, temp_path: Path):
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        
        temp_path.mkdir(parents=True, exist_ok=True)
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
        return {"glb_path": str(mesh_path)}

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"    
    sock_path = sys.argv[2]

    device = torch.device(device)
    generator = ModelGenerator(device)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    json_in = sock.makefile('r')
    json_out = sock.makefile('w')

    print(write_json({"status": "ready"}), file=json_out, flush=True)

    for line in json_in:
        request = parse_json(line.strip())
        if request["action"] == "meshify":
            result = generator.meshify(request["image_b64"], Path(request["temp_path"]))
            print(write_json(result), file=json_out, flush=True)
            clean_device_cache(device)
        elif request["action"] == "exit":
            break

    sock.close()