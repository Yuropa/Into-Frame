from util.image_utils import Image
from pathlib import Path

class ContextValue():
    def __init__(self, name) -> None:
        self.type = None
        self.value = None
        self.name = name

    def set_image(self, image):
        self.type = "image"
        self.value = Image(image)

    def image(self):
        if self.type == "image":
            return self.value
        else:
            return None
        
    def write(self, path):
        if self.type == "image":
            self.image().save(path=str(path / (self.name + ".png")))

class PipelineContext():
    def __init__(self, input: any) -> None:
        self.input_image = Image(input) 
        self._stage_state = {}
        self._state = {}
        self._current_stage = ""

    def push_stage(self, name: str):
        self._current_stage = name

    def pop_stage(self):
        self._current_stage = ""

    def _value(self, name):
        if len(self._current_stage) == 0:
            return self._state[name]
    
        if self._current_stage in self._stage_state:
            return self._stage_state[self._current_stage][name]
        
        self._stage_state[self._current_stage] = {}
        return self._stage_state[self._current_stage][name]

    def _set_value(self, name, value):
        if len(self._current_stage) == 0:
            self._state[name] = value
            return
    
        if self._current_stage in self._stage_state:
            self._stage_state[self._current_stage][name] = value
            return
        
        self._stage_state[self._current_stage] = {}
        self._stage_state[self._current_stage][name] = value

    def add_image(self, name: str, input: any):
        value = ContextValue(name=name)
        value.set_image(input)
        self._set_value(name, value)

    def image(self, name: str):
        return self._value(name).image()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True) 
        self.input_image.save(path=str(path / "input.png"))

        for value in self._state.values():
            value.write(path)

        for stage_name, values in self._stage_state.items():
            new_path = path / stage_name
            new_path.mkdir(parents=True, exist_ok=True) 

            for value in values.values():
                value.write(new_path)