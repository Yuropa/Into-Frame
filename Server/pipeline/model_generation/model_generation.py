import torch
import trimesh
import base64
from io import BytesIO
from pathlib import Path
import clip

# from transformers import Sam3Processor, Sam3Model

from scene.mesh import Mesh
from util.image_utils import Image
from util.json_utils import parse_json, write_json
from remote_connection.remote_client import RemoteClient

class ModelGenerator(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        clip.load('ViT-L/14@336px')
        script_path = Path(__file__).parent / "model_generation_spar3d_imp.py"

        super().__init__(
            device=device, 
            conda_env="stablepoint", 
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["stabilityai/stable-point-aware-3d"]
    
    def meshify(self, image: Image, temp_path: Path) -> Mesh:
        buffer = BytesIO()
        image.image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        response = self.send(action="meshify", input=image_b64, temp_path=temp_path)
        glb_path = Path(response)

        try:
            mesh = trimesh.load(str(glb_path))

            # trimesh.load returns a Scene for GLB — extract the geometry
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

            return Mesh(mesh)
        finally:
            glb_path.unlink(missing_ok=True)
    

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