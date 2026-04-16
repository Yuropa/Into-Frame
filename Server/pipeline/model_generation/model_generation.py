import torch
from transformers import Sam3Processor, Sam3Model
# from transparent_background import Remover
# from spar3d.system import SPAR3D
# from spar3d.utils import foreground_crop, remove_background

from scene.mesh import Mesh
from util.image_utils import Image

# class ModelGenerator_stable_point_3d():
#     def __init__(self, device) -> None:
#         self.device = device
#         self.image_resolution = 2024
#         self.foreground_ratio = 1.3
#         low_vram_mode=False
#         self.model = SPAR3D.from_pretrained(
#             "stabilityai/stable-point-aware-3d",
#             low_vram_mode=low_vram_mode
#         )
#         self.model.to(device)
#         self.model.eval()

#         self.bg_remove = Remover(device=device)

#     @classmethod
#     def model_names(cls) -> list[str]:
#         return ["stabilityai/stable-point-aware-3d"]
    
#     def meshify(self, image: Image):
#         cleaned_image = remove_background(
#             image.rgba(),
#             self.bg_remove
#         )
#         cleaned_image = foreground_crop(
#             cleaned_image,
#             self.foreground_ratio
#         )

#         mesh, glob_dict = self.model.run_image(
#             image,
#             bake_resolution=self.image_resolution,
#             remesh="triangle",
#             vertex_count=-1,
#             return_points=True,
#         )
        
#         return Mesh(mesh)
    

class ModelGenerator():
    def __init__(self, device) -> None:
        self.device = device
        self.model_id = "facebook/sam3"
        
        self.processor = Sam3Processor.from_pretrained(self.model_id)
        self.model = Sam3DModel.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 # Mandatory for 24GB Mac
        ).to(self.device)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["facebook/sam3"] # "facebook/sam-3d-objects"
    
    def meshify(
        self, 
        image: Image,
        prompt: str = "an object"
    ):
        raw_image = image.rgb()

        inputs = self.processor(raw_image, text=prompt, return_tensors="pt").to(self.device)
            
        with torch.no_grad():
            # SAM3DModel returns vertices, faces, and texture maps directly
            outputs = self.model.generate_mesh(**inputs)
            
        # 3. Build the Trimesh object
        # The .generate_mesh() call returns a dictionary of CPU tensors
        mesh = trimesh.Trimesh(
            vertices=outputs.vertices.numpy(),
            faces=outputs.faces.numpy(),
            process=True
        )
        
        # 4. Apply Texture (SAM 3D is famous for its high-quality baking)
        if hasattr(outputs, "textures"):
            mesh.visual = trimesh.visual.TextureVisuals(
                image=outputs.textures, # This is a PIL image
                uv=outputs.uvs.numpy()
            )
        
        return Mesh(mesh)