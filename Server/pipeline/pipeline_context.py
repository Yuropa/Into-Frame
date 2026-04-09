from util.image_utils import Image
from pathlib import Path
import json
import numpy as np

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
    def __init__(self, name) -> None:
        self.type = None
        self.value = None
        self.name = name

    def set_image(self, image):
        self.type = "image"
        self.value = Image(image)

    def set_object(self, obj):
        self.type = "obj"
        self.value = obj

    def image(self):
        if self.type == "image":
            return self.value
        else:
            return None
        
    def object(self):
        if self.type == "obj":
            return self.value
        else:
            return None
        
    def write(self, path):
        if self.type == "image":
            self.image().save(path=str(path / (self.name + ".png")))
        elif self.type == "obj":
            with open(str(path / (self.name + ".json")), "w") as f:
                json.dump(self.object(), f, indent=4, cls=JSONEncoder)

class PipelineContext():
    def __init__(self) -> None:
        self._stage_state = {}
        self._state = {}
        self._current_stage = ""
        self._previous_stage = ""

    def push_stage(self, name: str):
        self._current_stage = name

    def pop_stage(self):
        self._previous_stage = self._current_stage
        self._current_stage = ""

    def _value(self, name, search_stage = None):
        if search_stage is None:
            search_stage = self._current_stage

        if len(search_stage) == 0:
            return self._state[name]
    
        if search_stage in self._stage_state:
            return self._stage_state[search_stage][name]
        
        return None

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
    
    def input_image(self, name: str):
        return self._value(name, self._previous_stage).image()
    
    def add_object(self, name: str, input: any):
        value = ContextValue(name=name)
        value.set_object(input)
        self._set_value(name, value)

    def object(self, name: str):
        return self._value(name).object()
    
    def input_object(self, name: str):
        return self._value(name, self._previous_stage).object()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True) 
        for value in self._state.values():
            value.write(path)

        for stage_name, values in self._stage_state.items():
            new_path = path / stage_name
            new_path.mkdir(parents=True, exist_ok=True) 

            for value in values.values():
                value.write(new_path)