import math
import numpy as np

class CameraIntrinsics:
    def __init__(self, width: int, height: int, fov_degrees: float = 60.0):
        self.width  = width
        self.color_width = width
        self.height = height
        self.color_height = height
        self.fov    = fov_degrees

        self.px = width  / 2.0
        self.py = height / 2.0
        self.fx = (width  / 2.0) / math.tan(math.radians(fov_degrees / 2.0))
        self.fy = self.fx  # square pixels

    def unproject(self, cx: float, cy: float, depth: float) -> tuple[float, float, float]:
        X =  (cx - self.px) * depth / self.fx
        Y = -((cy - self.py) * depth / self.fy)  # flip Y for Unity
        Z =  depth
        return (X, Y, Z)

    def encode(self) -> dict:
        return {
            "width":    self.width,
            "height":   self.height,
            "fov":      self.fov,
            "px":       self.px,
            "py":       self.py,
            "fx":       self.fx,
            "fy":       self.fy
        }

    @classmethod
    def from_depth_anything(cls, intrinsics: np.ndarray, color_width: int, color_height: int, depth_width: int, depth_height: int):
        K = intrinsics[0]
        obj = cls.__new__(cls)
        obj.width  = depth_width
        obj.height = depth_height
        obj.color_width  = color_width
        obj.color_height = color_height
        obj.fx = float(K[0, 0])
        obj.fy = float(K[1, 1])
        obj.px = float(K[0, 2])
        obj.py = float(K[1, 2])
        obj.fov = math.degrees(2.0 * math.atan(depth_width / (2.0 * obj.fx)))
        return obj

