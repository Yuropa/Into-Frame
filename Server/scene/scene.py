
class Scene:
    def __init__(self):
        self.ambient_color = "#1a1a2e"
        self.fog_enabled = True
        self.fog_density = 0.02
        self.fog_color = "#0a0a1a"
        self.gravity = -9.81
        self.time_scale = 1.0

    def encode(self):
        return {
            "ambientColor": self.ambient_color,
            "fogEnabled":   self.fog_enabled,
            "fogDensity":   self.fog_density,
            "fogColor":     self.fog_color,
            "gravity":      self.gravity,
            "timeScale":    self.time_scale,
        }