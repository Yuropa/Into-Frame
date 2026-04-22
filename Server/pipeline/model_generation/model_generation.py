import torch
import trimesh
import numpy as np
import subprocess
import json
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

import trimesh
from trellis2.pipelines import Trellis2ImageTo3DPipeline

class ModelGenerator():
    def __init__(self, device) -> None:
        self.device = device
        self.model_id = "microsoft/TRELLIS.2-4B"

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_id)
        self.pipeline.cuda()

    @classmethod
    def model_names(cls) -> list[str]:
        return ["microsoft/TRELLIS.2-4B"]

    def meshify(
        self,
        image: Image,
        decimation_target: int = 1_000_000,
        texture_size: int = 4096,
    ):
        raw_image = image.rgb()

        # Run the image-to-3D pipeline
        mesh = self.pipeline.run(raw_image)[0]

        # Optional: simplify to stay within nvdiffrast's 16M face limit
        mesh.simplify(16_777_216)

        # Export to GLB via o_voxel postprocessing (handles PBR baking)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )

        # Convert the exported GLB into a trimesh for downstream compatibility
        tri_mesh = trimesh.load(glb, force="mesh")
        return Mesh(tri_mesh)

# class ModelGenerator():
#     def _pipe_stream(self, stream, prefix):
#         for line in stream:
#             print(f"[spar3d] {line}", end="", flush=True)
    
#     def __init__(self, device) -> None:
#         clip.load('ViT-L/14@336px')

#         self.device = device
#         self.process = None
#         script_path = Path(__file__).parent / "model_generation_spar3d_imp.py"

#         self.sock_path = tempfile.mktemp(suffix=".sock")
#         self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#         self.server_sock.bind(self.sock_path)
#         self.server_sock.listen(1)

#         self.process = subprocess.Popen(
#             ["conda", "run", "-n", "stablepoint", "python", str(script_path), str(device), self.sock_path],
#             stdin=subprocess.PIPE,
#             stdout=subprocess.PIPE, 
#             stderr=subprocess.PIPE,
#             text=False
#         )

#         threading.Thread(target=self._pipe_stream, args=(self.process.stdout, "[spar3d stdout]"), daemon=True).start()
#         threading.Thread(target=self._pipe_stream, args=(self.process.stderr, "[spar3d stderr]"), daemon=True).start()

#         self.server_sock.settimeout(60)
#         conn, _ = self.server_sock.accept()
#         self.json_pipe = conn.makefile('r')
#         self.json_out = conn.makefile('w')

#         ready_line = self.json_pipe.readline()
#         if not ready_line:
#             raise RuntimeError("spar3d subprocess produced no output")
#         ready = json.loads(ready_line)
#         if ready.get("status") != "ready":
#             raise RuntimeError(f"Unexpected startup message: {ready_line}")

#         print(f"Finished loading script {script_path}")

#     def __del__(self):
#         if self.process is not None:
#             self.close()

#     def _check_for_errors(self):
#         if self.process.poll() is not None:
#             stderr = self.process.stderr.read()
#             raise RuntimeError(f"spar3d subprocess exited before meshify:\n{stderr}")

#     @classmethod
#     def model_names(cls) -> list[str]:
#         return ["stabilityai/stable-point-aware-3d"]
    
#     def meshify(self, image: Image):
#         self._check_for_errors()

#         buffer = BytesIO()
#         image.image.save(buffer, format="PNG")
#         image_b64 = base64.b64encode(buffer.getvalue()).decode()

#         request = json.dumps({"action": "meshify", "image_b64": image_b64})
#         self.json_out.write(request + "\n")
#         self.json_out.flush()

#         response = json.loads(self.json_pipe.readline())
#         mesh = trimesh.Trimesh(
#             vertices=np.array(response["vertices"]),
#             faces=np.array(response["faces"])
#         )
#         return Mesh(mesh)
    
#     def close(self):
#         if self.process is not None:
#             try:
#                 request = json.dumps({"action": "exit"})
#                 self.json_out.write(request + "\n")
#                 self.json_out.flush()
#             except (BrokenPipeError, OSError):
#                 pass
#             self.process.wait()
#             self.server_sock.close()
#             Path(self.sock_path).unlink(missing_ok=True)
#             self.process = None
    

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