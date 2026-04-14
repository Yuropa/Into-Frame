import uuid
from enum import StrEnum

def vec3(x=0.0, y=0.0, z=0.0):
    return {"x": float(x), "y": float(y), "z": float(z)}

class ObjectType(StrEnum):
    BILLBOARD = "billboard"
    MESH = "mesh"

class Object3D():
    def __init__(self) -> None:
        self.id = uuid.uuid4()

        self.type = ""
        self.texture = ""
        self.position = vec3()
        self.rotation = vec3()
        self.scale = vec3(1.0, 1.0, 1.0)

    def set_position(self, x: float, y: float, z: float):
        self.position = vec3(x, y, z)

    def set_rotation(self, x: float, y: float, z: float):
        self.rotation = vec3(x, y, z)

    def set_scale(self, x: float, y: float, z: float):
        self.scale = vec3(x, y, z)
    
    def encode(self) -> dict:
        return {
            "id":       str(self.id),
            "position": self.position,
            "rotation": self.rotation,
            "scale":    self.scale,
            "type":     self.type,
            "texture":  self.texture
        }
    

    @classmethod
    def billboard(cls, texture: str, width: float, height: float, x: float, y: float, z: float):
        obj = cls()

        obj.type = ObjectType.BILLBOARD
        obj.texture = texture
        obj.set_position(x, y, z)
        obj.set_scale(width, height, 1.0)

        return obj