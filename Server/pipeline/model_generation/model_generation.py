import torch
import trimesh
import numpy as np
import subprocess
import base64
from io import BytesIO
import time
from pathlib import Path
import clip
import os
import threading
import socket
import tempfile

# from transformers import Sam3Processor, Sam3Model

from scene.mesh import Mesh
from util.image_utils import Image
from util.json_utils import parse_json, write_json

def readline_json(pipe):
    while True:
        line = pipe.readline()
        if not line:
            return None  # EOF
        if line.strip():
            return line.strip()

class ModelGenerator():
    def _pipe_stream(self, stream, prefix):
        for line in stream:        
            print(f"[spar3d] {line.decode('utf-8', errors='replace')}", end="", flush=True)

    
    def __init__(self, device) -> None:
        clip.load('ViT-L/14@336px')

        self.device = device
        self.process = None
        script_path = Path(__file__).parent / "model_generation_spar3d_imp.py"

        self.sock_path = tempfile.mktemp(suffix=".sock")
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.sock_path)
        self.server_sock.listen(1)

        self.process = subprocess.Popen(
            ["conda", "run", "--no-capture-output", "-n", "stablepoint", "python", str(script_path), str(device), self.sock_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=False
        )

        threading.Thread(target=self._pipe_stream, args=(self.process.stderr, "err"), daemon=True).start()
        threading.Thread(target=self._pipe_stream, args=(self.process.stdout, "out"), daemon=True).start()

        self.server_sock.settimeout(60)
        try:
            conn, _ = self.server_sock.accept()
        except socket.timeout:
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                raise RuntimeError(f"Subprocess died while waiting:\n{stderr}")
            raise RuntimeError("Timed out waiting for subprocess to connect")

        self.json_pipe = conn.makefile('r')
        self.json_out = conn.makefile('w')

        ready_line = readline_json(self.json_pipe)
        if not ready_line:
            raise RuntimeError("spar3d subprocess produced no output")

        ready = parse_json(ready_line)
        print(f"Got ready line repr: {repr(ready_line)}")
        print(f"Got ready line {ready_line} and obj {ready}")
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected startup message: {ready_line}")

        print(f"Finished loading script {script_path}")

    def __del__(self):
        if self.process is not None:
            self.close()

    def _check_for_errors(self):
        if self.process.poll() is not None:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"spar3d subprocess exited before meshify:\n{stderr}")

    @classmethod
    def model_names(cls) -> list[str]:
        return ["stabilityai/stable-point-aware-3d"]
    
    def meshify(self, image: Image, temp_path: Path):
        self._check_for_errors()

        buffer = BytesIO()
        image.image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        request = write_json({"action": "meshify", "image_b64": image_b64, "temp_path": str(temp_path)})
        self.json_out.write(request)
        self.json_out.flush()

        response = parse_json(readline_json(self.json_pipe))

        glb_path = Path(response["glb_path"])
        mesh = trimesh.load(str(glb_path))

        # trimesh.load returns a Scene for GLB — extract the geometry
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        return Mesh(mesh)
    
    def close(self):
        if self.process is not None:
            try:
                if hasattr(self, 'json_out'):
                    request = write_json({"action": "exit"})
                    self.json_out.write(request)
                    self.json_out.flush()
            except (BrokenPipeError, OSError):
                pass
            self.process.wait()
            if hasattr(self, 'server_sock'):
                self.server_sock.close()
            if hasattr(self, 'sock_path'):
                Path(self.sock_path).unlink(missing_ok=True)
            self.process = None
    

# class ModelGenerator():
#     def __init__(self, device) -> None:
#         self.device = device
#         self.model_id = "facebook/sam3"
        
#         self.processor = Sam3Processor.from_pretrained(self.model_id)
#         self.model = Sam3DModel.from_pretrained(
#             self.model_id, 
#             torch_dtype=torch.float16 # Mandatory for 24GB Mac
#         ).to(self.device)

#     @classmethod
#     def model_names(cls) -> list[str]:
#         return ["facebook/sam3"] # "facebook/sam-3d-objects"
    
#     def meshify(
#         self, 
#         image: Image,
#         prompt: str = "an object"
#     ):
#         raw_image = image.rgb()

#         inputs = self.processor(raw_image, text=prompt, return_tensors="pt").to(self.device)
            
#         with torch.no_grad():
#             # SAM3DModel returns vertices, faces, and texture maps directly
#             outputs = self.model.generate_mesh(**inputs)
            
#         # 3. Build the Trimesh object
#         # The .generate_mesh() call returns a dictionary of CPU tensors
#         mesh = trimesh.Trimesh(
#             vertices=outputs.vertices.numpy(),
#             faces=outputs.faces.numpy(),
#             process=True
#         )
        
#         # 4. Apply Texture (SAM 3D is famous for its high-quality baking)
#         if hasattr(outputs, "textures"):
#             mesh.visual = trimesh.visual.TextureVisuals(
#                 image=outputs.textures, # This is a PIL image
#                 uv=outputs.uvs.numpy()
#             )
        
#         return Mesh(mesh)