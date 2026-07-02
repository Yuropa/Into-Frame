from scene.object import Object3D
from scene.camera import CameraExtrinsics
from typing import Self, Optional

class Scene:
    def __init__(self):
        self.ambient_color = "#ffffff"
        self.gravity = -9.81
        self.time_scale = 1.0
        self.near_clip_plane = 0.1
        self.far_clip_plane = 100.0
        self.skybox_rotation = 0.0
        self.camera_height = 0.0  # eye height above ground in metres; 0 = not set
        self.eye_height_meters = 1.8  # target world-space depth of the terrain center below the viewer
        self.terrain_center_y = 0.0   # world-space Y of the terrain surface directly below the capture point
        self.extrinsics = CameraExtrinsics.identity()
        self.objects = []
        self.skybox = ""
        self.lighting = None  # Optional[SceneLighting]

    def add_object(self, object: Object3D):
        self.objects.append(object)

    def encode(self):
        return {
            "ambientColor":  self.ambient_color,
            "gravity":       self.gravity,
            "timeScale":     self.time_scale,
            "nearClipPlane": self.near_clip_plane,
            "farClipPlane":  self.far_clip_plane,
            "objects":       [obj.encode() for obj in self.objects],
            "extrinsics":    self.extrinsics.encode(),
            "skybox":         self.skybox,
            "skyboxRotation": self.skybox_rotation,
            "cameraHeight":   self.camera_height,
            "eyeHeightMeters": self.eye_height_meters,
            "terrainCenterY":  self.terrain_center_y,
            "lighting":       self.lighting.encode() if self.lighting is not None else None,
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        from scene.lighting import SceneLighting
        obj = cls()
        obj.ambient_color   = data["ambientColor"]
        obj.gravity         = data["gravity"]
        obj.time_scale      = data["timeScale"]
        obj.near_clip_plane = data.get("nearClipPlane", 0.1)
        obj.far_clip_plane  = data.get("farClipPlane", 100.0)
        obj.extrinsics      = CameraExtrinsics.decode(data["extrinsics"])
        obj.objects         = [Object3D.decode(o) for o in data["objects"]]
        obj.skybox          = data["skybox"]
        obj.skybox_rotation = data.get("skyboxRotation", 0.0)
        obj.camera_height   = data.get("cameraHeight", 0.0)
        obj.eye_height_meters = data.get("eyeHeightMeters", 1.8)
        obj.terrain_center_y  = data.get("terrainCenterY", 0.0)
        lighting_data       = data.get("lighting")
        obj.lighting        = SceneLighting.decode(lighting_data) if lighting_data else None
        return obj