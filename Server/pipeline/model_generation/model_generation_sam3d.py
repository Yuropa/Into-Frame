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