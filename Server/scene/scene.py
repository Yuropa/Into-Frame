from scene.object import Object3D
from scene.camera import CameraExtrinsics
from typing import Self

class Scene:
    def __init__(self):
        self.ambient_color = "#1a1a2e"
        self.gravity = -9.81
        self.time_scale = 1.0
        self.extrinsics = CameraExtrinsics.identity()
        self.objects = []
        self.skybox = ""

    def add_object(self, object: Object3D):
        self.objects.append(object)

    def encode(self):
        return {
            "ambientColor": self.ambient_color,
            "gravity":      self.gravity,
            "timeScale":    self.time_scale,    
            "objects":      [obj.encode() for obj in self.objects],
            "extrinsics":   self.extrinsics.encode(),
            "skybox":       self.skybox,
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        obj = cls()
        obj.ambient_color = data["ambientColor"]
        obj.gravity = data["gravity"]
        obj.time_scale = data["timeScale"]
        obj.extrinsics = CameraExtrinsics.decode(data["extrinsics"])
        obj.objects = [Object3D.decode(o) for o in data["objects"]]
        obj.skybox = data["skybox"]
        return obj