import json
import numpy as np
from util.image_utils import Image
from util.depth_utils import Depth
from pathlib import Path
from typing import Optional, Any
from enum import StrEnum
from scene.mesh import Mesh
from scene.scene import Scene
from scene.object import Object3D
from scene.camera import CameraIntrinsics

class ValueKeys(StrEnum):
    NONE = "none"
    IMAGE = "input"
    MESH = "mesh"
    OBJECT = "object"
    DEPTH = "depth"
    OBJECT3D = "object_3d"
    SCENE = "scene"
    INTRINSICS = "intrinsics"

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
        
    def write(self, path: Path):
        if self.type == ValueKeys.IMAGE:
            self.image().save(path=str(path / (self.name + ".png")))
        elif self.type == ValueKeys.MESH:
            self.mesh().save(str(path / (self.name + ".glb")))
        elif self.type == ValueKeys.OBJECT:
            with open(str(path / (self.name + ".json")), "w") as f:
                json.dump(self.object(), f, indent=4, cls=JSONEncoder)
        elif self.type == ValueKeys.DEPTH:
            self.depth().save(path=str(path / (self.name + ".png")))
        elif self.type == ValueKeys.OBJECT3D:
            with open(str(path / (self.name + ".json")), "w") as f:
                json.dump(self.object3d().encode(), f, indent=4, cls=JSONEncoder)
        elif self.type == ValueKeys.SCENE:
            with open(str(path / (self.name + ".json")), "w") as f:
                json.dump(self.scene().encode(), f, indent=4, cls=JSONEncoder)
        elif self.type == ValueKeys.INTRINSICS:
            with open(str(path / (self.name + ".json")), "w") as f:
                json.dump(self.intrinsics().encode(), f, indent=4, cls=JSONEncoder)