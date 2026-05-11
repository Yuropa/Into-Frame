import json
import numpy as np
from util.image_utils import Image
from util.depth_utils import Depth
from util.cubemap_utils import CubeMap
from pathlib import Path
from typing import Optional, Any
from enum import StrEnum
from scene.mesh import Mesh
from scene.scene import Scene
from scene.object import Object3D
from scene.camera import CameraIntrinsics, CameraExtrinsics

class ValueKeys(StrEnum):
    NONE = "none"
    IMAGE = "image"
    MESH = "mesh"
    OBJECT = "object"
    DEPTH = "depth"
    OBJECT3D = "object_3d"
    SCENE = "scene"
    INTRINSICS = "intrinsics"
    EXTRINSICS = "extrinsics"
    CUBEMAP = "cubemap"

    def preferred_extension(self) -> "str":
        match self:
            case ValueKeys.IMAGE:
                return "png"
            case ValueKeys.MESH:
                return "glb"
            case ValueKeys.DEPTH:
                return "npy"
            case ValueKeys.OBJECT:
                return "json"
            case ValueKeys.OBJECT3D:
                return "json"
            case ValueKeys.SCENE:
                return "json"
            case ValueKeys.INTRINSICS:
                return "json"
            case ValueKeys.EXTRINSICS:
                return "json"
            case ValueKeys.CUBEMAP:
                return "cube"

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class ContextValue():
    name: str
    type: ValueKeys
    value: Optional[Any]

    def __init__(self, name) -> None:
        self.type = ValueKeys.NONE
        self.value = None
        self.name = name

    def set_image(self, image: Any):
        self.type = ValueKeys.IMAGE
        self.value = Image(image)

    def set_mesh(self, mesh: Mesh):
        self.type = ValueKeys.MESH
        self.value = mesh

    def set_object(self, obj: Any):
        self.type = ValueKeys.OBJECT
        self.value = obj

    def set_depth(self, obj: Any):
        self.type = ValueKeys.DEPTH
        self.value = Depth(obj)

    def set_object3d(self, obj: Object3D):
        self.type = ValueKeys.OBJECT3D
        self.value = obj

    def set_scene(self, obj: Scene):  
        self.type = ValueKeys.SCENE
        self.value = obj  

    def set_intrinsics(self, obj: CameraIntrinsics):  
        self.type = ValueKeys.INTRINSICS
        self.value = obj

    def set_extrinsics(self, obj: CameraExtrinsics):  
        self.type = ValueKeys.EXTRINSICS
        self.value = obj

    def set_cubemap(self, obj: CubeMap):  
        self.type = ValueKeys.CUBEMAP
        self.value = obj

    def image(self) -> Optional[Image]:
        if self.type == ValueKeys.IMAGE:
            return self.value
        else:
            return None
        
    def object(self) -> Optional[Any]:
        if self.type == ValueKeys.OBJECT:
            return self.value
        else:
            return None
        
    def mesh(self) -> Optional[Mesh]:
        if self.type == ValueKeys.MESH:
            return self.value
        else:
            return None
        
    def depth(self) -> Optional[Depth]:
        if self.type == ValueKeys.DEPTH:
            return self.value
        else:
            return None
        
    def object3d(self) -> Optional[Object3D]:
        if self.type == ValueKeys.OBJECT3D:
            return self.value
        else:
            return None
        
    def scene(self) -> Optional[Scene]:
        if self.type == ValueKeys.SCENE:
            return self.value
        else:
            return None
        
    def intrinsics(self) -> Optional[CameraIntrinsics]:
        if self.type == ValueKeys.INTRINSICS:
            return self.value
        else:
            return None
        
    def extrinsics(self) -> Optional[CameraExtrinsics]:
        if self.type == ValueKeys.EXTRINSICS:
            return self.value
        else:
            return None
        
    def cubemap(self) -> Optional[CubeMap]:
        if self.type == ValueKeys.CUBEMAP:
            return self.value
        else:
            return None

    def read(self, path: Path):
        meta_path = path / (self.name + ".meta")
        if not meta_path.exists():
            return
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        value_type = ValueKeys(meta["type"])
        resolved_path = path / (self.name + "." + value_type.preferred_extension())

        if value_type == ValueKeys.IMAGE:
            self.set_image(Image.load(resolved_path))
        elif value_type == ValueKeys.MESH:
            self.set_mesh(Mesh.load(resolved_path))
        elif value_type == ValueKeys.DEPTH:
            self.set_depth(Depth.load(resolved_path))
        elif value_type == ValueKeys.OBJECT:
            with open(resolved_path) as f:
                self.set_object(json.load(f))
        elif value_type == ValueKeys.OBJECT3D:
            with open(resolved_path) as f:
                self.set_object3d(Object3D.decode(json.load(f)))
        elif value_type == ValueKeys.SCENE:
            with open(resolved_path) as f:
                self.set_scene(Scene.decode(json.load(f)))
        elif value_type == ValueKeys.INTRINSICS:
            with open(resolved_path) as f:
                self.set_intrinsics(CameraIntrinsics.decode(json.load(f)))
        elif value_type == ValueKeys.EXTRINSICS:
            with open(resolved_path) as f:
                self.set_extrinsics(CameraExtrinsics.decode(json.load(f)))
        elif value_type == ValueKeys.CUBEMAP:
            self.set_cubemap(CubeMap.load(resolved_path))
        
    def write(self, path: Path) -> Path:
        meta_path = path / (self.name + ".meta")
        with open(meta_path, "w") as f:
            json.dump({"type": self.type}, f)
        
        save_path = path / (self.name + "." + self.type.preferred_extension())

        if self.type == ValueKeys.IMAGE:
            self.image().save(path=save_path)
        elif self.type == ValueKeys.MESH:
            self.mesh().save(path=save_path)
        elif self.type == ValueKeys.OBJECT:
            with open(save_path, "w") as f:
                json.dump(self.object(), f, indent=4, cls=JSONEncoder)
        elif self.type == ValueKeys.DEPTH:
            self.depth().save(path=save_path)
        elif self.type == ValueKeys.OBJECT3D:
            with open(save_path, "w") as f:
                json.dump(self.object3d().encode(), f, indent=4, cls=JSONEncoder)
        elif self.type == ValueKeys.SCENE:
            with open(save_path, "w") as f:
                json.dump(self.scene().encode(), f, indent=4, cls=JSONEncoder)
        elif self.type == ValueKeys.INTRINSICS:
            with open(save_path, "w") as f:
                json.dump(self.intrinsics().encode(), f, indent=4, cls=JSONEncoder)
        elif self.type == ValueKeys.EXTRINSICS:
            with open(save_path, "w") as f:
                json.dump(self.extrinsics().encode(), f, indent=4, cls=JSONEncoder)
        elif self.type == ValueKeys.CUBEMAP:
            self.cubemap().save(path=save_path)

        return save_path
    
    def describe(self) -> str:
        if self.type == ValueKeys.IMAGE:
            v = self.image()
            return f"Image ({v.width}x{v.height})"
        elif self.type == ValueKeys.DEPTH:
            v = self.depth()
            return f"Depth ({v.width}x{v.height}, {v.min():.2f}–{v.max():.2f})"
        elif self.type == ValueKeys.MESH:
            v = self.mesh()
            return f"Mesh ({v.vertex_count} verts, {v.face_count} faces)"
        elif self.type == ValueKeys.SCENE:
            v = self.scene()
            return f"Scene ({len(v.objects)} objects)"
        elif self.type == ValueKeys.OBJECT3D:
            return f"Object3D"
        elif self.type == ValueKeys.INTRINSICS:
            v = self.intrinsics()
            return f"Intrinsics ({v.width}x{v.height}, fov={v.fov:.1f}°)"
        elif self.type == ValueKeys.EXTRINSICS:
            return f"Extrinsics"
        elif self.type == ValueKeys.OBJECT:
            return f"Object ({type(self.object()).__name__})"
        elif self.type == ValueKeys.CUBEMAP:
            return f"CubeMap ({self.cubemap().type.value})"
        return "None"
