import uuid
from enum import StrEnum
from typing import Optional, Self

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
        self.mesh = ""
        # Optional separate, lower-poly mesh asset key Unity loads as a
        # MeshCollider on this object -- e.g. the terrain's TERRAIN_PHYSICS_MESH,
        # a coarse geometry-only proxy instead of the dense, textured render
        # mesh. Empty (the default, and every non-terrain mesh today) -> no
        # collider is attached, same as before this field existed.
        self.physics_mesh = ""
        self.name = ""
        self.position = vec3()
        self.rotation = vec3()
        self.scale = vec3(1.0, 1.0, 1.0)

        # Join key back to the detection this object came from (SceneGenerationStage
        # sets it for both the billboard and per-instance mesh placement branches).
        # Exists so a later pass (SceneAnimationStage) can look up that detection's
        # own object_motion_{idx}/metadata_{idx} even when several instances share
        # one rendered mesh asset (see PanoramaAssetGenerationStage's
        # category_mesh_{class}_{bucket}). Not used by the Unity client, but still encoded --
        # a cached/resumed run reloads Scene Generation's output from its persisted
        # JSON (Object3D.decode()) rather than keeping the live Python objects
        # around, so this has to round-trip through JSON or a resumed run would
        # silently lose every object's join key and animate nothing.
        self.source_index: Optional[int] = None

        # Animated-billboard video assets (asset keys, same convention as
        # `texture`), set by SceneAnimationStage for a stationary billboard with
        # a corresponding extracted clip. None/empty -> render the static
        # `texture` as before.
        self.video_color: str = ""
        self.video_alpha: str = ""

        # Gentle procedural sway for a stationary mesh instance (wind on
        # foliage), e.g. {"amplitude": 0.05, "frequencyHz": 0.3, "phase": 1.2,
        # "axisDegrees": 40.0}. Set by SceneAnimationStage; consumed at runtime
        # by Unity's WindSway component against the bones CategoryMeshRiggingStage
        # baked into the mesh asset. None when not applicable.
        self.sway: Optional[dict] = None

        # Rigid-body handoff for a detected moving object, e.g. {"velocity": [x,y,z],
        # "acceleration": [x,y,z], "colliderShape": "box"|"capsule", "colliderSize":
        # [x,y,z]}. Set by SceneAnimationStage; Unity's PhysicsHandoff component
        # applies it once at spawn and lets its own Rigidbody own the motion from
        # then on. None when not applicable.
        self.physics: Optional[dict] = None

    def set_position(self, x: float, y: float, z: float):
        self.position = vec3(x, y, z)

    def set_rotation(self, x: float, y: float, z: float):
        self.rotation = vec3(x, y, z)

    def set_scale(self, x: float, y: float, z: float):
        self.scale = vec3(x, y, z)
    
    def encode(self) -> dict:
        data = {
            "id":       str(self.id),
            "position": self.position,
            "rotation": self.rotation,
            "scale":    self.scale,
            "type":     self.type,
            "texture":  self.texture,
            "mesh":     self.mesh,
            "name":     self.name
        }
        if self.source_index is not None:
            data["sourceIndex"] = self.source_index
        # Only present on animated objects -- keeps every untouched object's
        # payload identical to before this field set existed.
        if self.video_color:
            data["videoColor"] = self.video_color
        if self.video_alpha:
            data["videoAlpha"] = self.video_alpha
        if self.sway is not None:
            data["sway"] = self.sway
        if self.physics is not None:
            data["physics"] = self.physics
        if self.physics_mesh:
            data["physicsMesh"] = self.physics_mesh
        return data

    @classmethod
    def billboard(cls, texture: str, width: float, height: float, x: float, y: float, z: float):
        obj = cls()

        obj.type = ObjectType.BILLBOARD
        obj.texture = texture
        obj.set_position(x, y, z)
        obj.set_scale(width, height, 1.0)

        return obj

    @classmethod
    def mesh(cls, mesh: str, x: float, y: float, z: float):
        obj = cls()

        obj.type = ObjectType.MESH
        obj.mesh = mesh
        obj.set_position(x, y, z)

        return obj

    @classmethod
    def decode(cls, data: dict) -> Self:
        obj = cls()
        obj.id = uuid.UUID(data["id"])
        obj.type = data["type"]
        obj.texture = data["texture"]
        obj.mesh = data["mesh"]
        obj.position = data["position"]
        obj.rotation = data["rotation"]
        obj.scale = data["scale"]
        obj.name = data["name"]
        obj.source_index = data.get("sourceIndex")
        obj.video_color = data.get("videoColor", "")
        obj.video_alpha = data.get("videoAlpha", "")
        obj.sway = data.get("sway")
        obj.physics = data.get("physics")
        obj.physics_mesh = data.get("physicsMesh", "")
        return obj