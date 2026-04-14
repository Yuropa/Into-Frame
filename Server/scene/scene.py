
class Scene:
    def __init__(self):
        self.ambient_color = "#1a1a2e"
        self.gravity = -9.81
        self.time_scale = 1.0

    def encode(self):
        return {
            "ambientColor": self.ambient_color,
            "gravity":      self.gravity,
            "timeScale":    self.time_scale,
        }