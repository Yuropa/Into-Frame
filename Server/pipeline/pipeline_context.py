from pipeline.context_value import ContextValue
from pathlib import Path
from typing import Literal, TypeAlias, Optional, Any
from scene.mesh import Mesh
from util.depth_utils import Depth
from util.image_utils import Image

class ContextKey:
    INPUT = "input"
    DEPTH = "depth"
    Type = Literal["input", "depth"]

ContextKeyName: TypeAlias = ContextKey.Type | str

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

    def _value(self, name: ContextKeyName, search_stage: Optional[str] = None) -> ContextValue:
        if search_stage is None:
            search_stage = self._current_stage

        if len(search_stage) == 0:
            return self._state[name]
    
        if search_stage in self._stage_state:
            if name in self._stage_state[search_stage]:
                return self._stage_state[search_stage][name]
        
        if name in self._state:
            return self._state[name]

        return ContextValue("")

    def _set_value(self, name: ContextKeyName, value: ContextValue):
        if len(self._current_stage) == 0:
            self._state[name] = value
            return
    
        if self._current_stage in self._stage_state:
            self._stage_state[self._current_stage][name] = value
            return
        
        self._stage_state[self._current_stage] = {}
        self._stage_state[self._current_stage][name] = value

    # Image
    def add_image(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_image(input)
        self._set_value(name, value)

    def image(self, name: ContextKeyName) -> Optional[Image]:
        return self._value(name).image()
    
    def input_image(self, name: ContextKeyName) -> Optional[Image]:
        return self._value(name, self._previous_stage).image()
    
    # Object
    def add_object(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_object(input)
        self._set_value(name, value)

    def object(self, name: ContextKeyName) -> Optional[Any]:
        return self._value(name).object()
    
    def input_object(self, name: ContextKeyName) -> Optional[Any]:
        return self._value(name, self._previous_stage).object()

    # Mesh 
    def add_mesh(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_mesh(input)
        self._set_value(name, value)

    def mesh(self, name: ContextKeyName) -> Optional[Mesh]:
        return self._value(name).mesh()
    
    def input_mesh(self, name: ContextKeyName) -> Optional[Mesh]:
        return self._value(name, self._previous_stage).mesh()

    # Depth
    def add_depth(self, name: ContextKeyName, input: Any):
        value = ContextValue(name=name)
        value.set_mesh(input)
        self._set_value(name, value)

    def depth(self, name: ContextKeyName) -> Optional[Depth]:
        return self._value(name).depth()
    
    def input_depth(self, name: ContextKeyName) -> Optional[Depth]:
        return self._value(name, self._previous_stage).depth()
    
    # Persistence
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True) 
        for value in self._state.values():
            value.write(path)

        for stage_name, values in self._stage_state.items():
            new_path = path / stage_name
            new_path.mkdir(parents=True, exist_ok=True) 

            for value in values.values():
                value.write(new_path)