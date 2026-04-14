from scene.object import Object3D

class Scene:
    def __init__(self):
        self.ambient_color = "#1a1a2e"
        self.gravity = -9.81
        self.time_scale = 1.0
        self.objects = []

    def add_object(self, object: Object3D):
        self.objects.append(object)

    def encode(self):
        return {
            "ambientColor": self.ambient_color,
            "gravity":      self.gravity,
            "timeScale":    self.time_scale,    
            "objects":      [obj.encode() for obj in self.objects],
        }