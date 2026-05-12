from pipeline.context_value import ContextValue, ValueKeys
from pathlib import Path
from typing import Literal, TypeAlias, Optional, Any
from scene.mesh import Mesh
from util.depth_utils import Depth
from util.image_utils import Image
from util.cubemap_utils import CubeMap
from scene.scene import Scene
from scene.object import Object3D
from scene.camera import CameraIntrinsics, CameraExtrinsics

class ContextKey:
    INPUT = "input"
    DEPTH = "depth"
    SCENE = "scene"
    INTRINSICS = "intrinsics"
    EXTRINSICS = "extrinsics"
    PANORAMA = "panorama"
    INPUT_CAPTION = "input_caption"
    PANAORAMA_CUBENAME = "panorama_cubemap"
    OBJECT_COUNT = "count"
    FOREGROUND_MASKED_IMAGE = "foreground_masked_image"
    Type = Literal["input", "depth", "scene", "intrinsics", "panorama", "input_caption", "panorama_cubemap", "count", "foreground_masked_image"]

ContextKeyName: TypeAlias = ContextKey.Type | str

class PipelineContext():
    def __init__(self) -> None:
        self._stage_state = {}
        self._state = {}
        self._current_stage = ""
        self._previous_stage = ""
        self._stage_order = []
        self._dirty_state: set[str] = set()
        self._dirty_stage_state: dict[str, set[str]] = {}

    def push_stage(self, name: str):
        self._current_stage = name
        if name not in self._stage_order:
            self._stage_order.append(name)

    def pop_stage(self):
        self._previous_stage = self._current_stage
        self._current_stage = ""

    def _value(self, name: ContextKeyName, search_stage: Optional[str] = None) -> ContextValue:
        if search_stage is None:
            search_stage = self._current_stage

        # Build the search list: all stages up to and including search_stage, in reverse
        if search_stage and search_stage in self._stage_order:
            idx = self._stage_order.index(search_stage)
            stages_to_search = list(reversed(self._stage_order[:idx + 1]))
        else:
            stages_to_search = list(reversed(self._stage_order))

        # Walk stages in reverse order looking for the value
        for stage in stages_to_search:
            if stage in self._stage_state and name in self._stage_state[stage]:
                return self._stage_state[stage][name]

        # Fall back to global state
        if name in self._state:
            return self._state[name]

        return ContextValue("")

    def _set_value(self, name: ContextKeyName, value: ContextValue):
        if len(self._current_stage) == 0:
            self._state[name] = value
            self._dirty_state.add(name)
            return
    
        if self._current_stage not in self._stage_state:
            self._stage_state[self._current_stage] = {}
        if self._current_stage not in self._dirty_stage_state:
            self._dirty_stage_state[self._current_stage] = set()

        self._stage_state[self._current_stage][name] = value
        self._dirty_stage_state[self._current_stage].add(name)

    # Type
    def type_for(self, name: ContextKeyName) -> Optional[ValueKeys]:
        return self._value(name).type

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
        value.set_depth(input)
        self._set_value(name, value)

    def depth(self, name: ContextKeyName) -> Optional[Depth]:
        return self._value(name).depth()
    
    def input_depth(self, name: ContextKeyName) -> Optional[Depth]:
        return self._value(name, self._previous_stage).depth()
    
    # Object3D
    def add_object3d(self, name: ContextKeyName, input: Object3D):
        value = ContextValue(name=name)
        value.set_object3d(input)
        self._set_value(name, value)

    def object3d(self, name: ContextKeyName) -> Optional[Object3D]:
        return self._value(name).object3d()
    
    def input_object3d(self, name: ContextKeyName) -> Optional[Object3D]:
        return self._value(name, self._previous_stage).object3d()

    # Scene
    def add_scene(self, name: ContextKeyName, input: Scene):
        value = ContextValue(name=name)
        value.set_scene(input)
        self._set_value(name, value)

    def scene(self, name: ContextKeyName) -> Optional[Scene]:
        return self._value(name).scene()
    
    def input_scene(self, name: ContextKeyName) -> Optional[Scene]:
        return self._value(name, self._previous_stage).scene()
    
    # Intrinsics
    def add_intrinsics(self, name: ContextKeyName, input: CameraIntrinsics):
        value = ContextValue(name=name)
        value.set_intrinsics(input)
        self._set_value(name, value)

    def intrinsics(self, name: ContextKeyName) -> Optional[CameraIntrinsics]:
        return self._value(name).intrinsics()
    
    def input_intrinsics(self, name: ContextKeyName) -> Optional[CameraIntrinsics]:
        return self._value(name, self._previous_stage).intrinsics()
    
    # Extrinsics
    def add_extrinsics(self, name: ContextKeyName, input: CameraExtrinsics):
        value = ContextValue(name=name)
        value.set_extrinsics(input)
        self._set_value(name, value)

    def extrinsics(self, name: ContextKeyName) -> Optional[CameraExtrinsics]:
        return self._value(name).extrinsics()
    
    def input_extrinsics(self, name: ContextKeyName) -> Optional[CameraExtrinsics]:
        return self._value(name, self._previous_stage).extrinsics()

    # CubeMap
    def add_cubemap(self, name: ContextKeyName, input: CubeMap):
        value = ContextValue(name=name)
        value.set_cubemap(input)
        self._set_value(name, value)

    def cubemap(self, name: ContextKeyName) -> Optional[CubeMap]:
        return self._value(name).cubemap()
    
    def input_cubemap(self, name: ContextKeyName) -> Optional[CubeMap]:
        return self._value(name, self._previous_stage).cubemap()
    
    # Persistence
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for name in self._dirty_state:
            self._state[name].write(path)
        self._dirty_state.clear()

        for stage_name, dirty_keys in self._dirty_stage_state.items():
            stage_path = path / stage_name
            stage_path.mkdir(parents=True, exist_ok=True)
            for name in dirty_keys:
                self._stage_state[stage_name][name].write(stage_path)
        self._dirty_stage_state.clear()                 

    def save_object(self, name: ContextKeyName, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True) 
        return self._value(name).write(path=path)

    def load(self, path: Path, stage_order: list[str]):
        if not path.exists():
            return
        self._stage_order = stage_order
        self._load_directory(path, self._state)
        for stage_path in sorted(path.iterdir()):
            if stage_path.is_dir():
                stage_name = stage_path.name
                if stage_name not in self._stage_state:
                    self._stage_state[stage_name] = {}
                if stage_name not in self._stage_order:
                    self._stage_order.append(stage_name)
                self._load_directory(stage_path, self._stage_state[stage_name])

    def _load_directory(self, path: Path, target: dict):
        for meta_file in path.glob("*.meta"):
            name = meta_file.stem
            try:
                value = ContextValue(name)
                value.read(path)
                target[name] = value
            except Exception as e:
                print(f"Skipping '{name}' in {path}: {e}")

    def log_state(self):
        def _print_values(values: dict, indent: str):
            items = sorted(values.items())
            for i, (name, value) in enumerate(items):
                connector = "└──" if i == len(items) - 1 else "├──"
                print(f"{indent}{connector} {name}: {value.describe()}")

        print("\n PipelineContext")
        if self._state:
            has_stages = bool(self._stage_state)
            connector = "├──" if has_stages else "└──"
            print(f" {connector} [global]")
            _print_values(self._state, " │   " if has_stages else "     ")

        stages = list(self._stage_state.items())
        for i, (stage_name, values) in enumerate(stages):
            connector = "└──" if i == len(stages) - 1 else "├──"
            print(f" {connector} [{stage_name}]")
            _print_values(values, "     " if i == len(stages) - 1 else " │   ")

        print()
        