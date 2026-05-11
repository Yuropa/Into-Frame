from __future__ import annotations
from util.image_utils import Image
from util.depth_utils import Depth
from util.json_utils import write_json, parse_json
from enum import StrEnum, Enum
from typing import Optional, TypeVar, Callable
from pathlib import Path
import PIL

class CubeFace(StrEnum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

    @classmethod
    def best_match(cls, name: str) -> Optional[CubeFace]:
        name = name.lower().strip()
        if not name:
            return None

        try:
            return cls(name)
        except ValueError:
            pass

        for face in cls:
            if face.value[0] == name[0]:
                return face

        return None
    
class CubeMapType(StrEnum):
    IMAGE = "image"
    DEPTH = "depth"

    def preferred_extension(self) -> "str":
        match self:
            case CubeMapType.IMAGE:
                return "png"
            case CubeMapType.DEPTH:
                return "npy"
            
    @classmethod
    def preferred_type(self, obj) -> Optional[CubeMapType]:
        if isinstance(obj, Image):
            return CubeMapType.IMAGE
        elif isinstance(obj, PIL.Image.Image):
            return CubeMapType.IMAGE
        elif isinstance(obj, Depth):
            return CubeMapType.DEPTH
        else:
            None

class CubeMap:
    images: dict[CubeFace, Image | Depth]
    type: CubeMapType
    REQUIRED_FACES = set(CubeFace)
    META_FILE = "cubemap.json"

    def __init__(self, obj: dict | Path | str, cube_type: CubeMapType | None = None):
        if isinstance(obj, str):
            obj = Path(obj)

        if isinstance(obj, Path):
            loaded = CubeMap.load(obj)
            self.images = loaded.images
            self.type = loaded.type
            return
        elif isinstance(obj, CubeMap):
            self.images = obj.images
            self.type = obj.type
            return

        if not isinstance(obj, dict):
            raise TypeError(f"Unsupported type: {type(obj)}")

        parsed_objects = {}
        for key, value in obj.items():
            face = CubeFace.best_match(key) if isinstance(key, str) else key
            if face is not None:
                parsed_objects[face] = value

        if cube_type is None:
            first = next(iter(parsed_objects.values()), None)
            if first is None:
                raise ValueError("Cannot infer CubeMapType from empty dict")
            matched = CubeMapType.preferred_type(first)
            if matched is None:
                raise TypeError(f"Cannot infer CubeMapType from {type(first)}")
            self.type = matched
        else:
            self.type = cube_type

        self.images = {}
        for face, value in parsed_objects.items():
            match self.type:
                case CubeMapType.IMAGE:
                    self.images[face] = Image(value)
                case CubeMapType.DEPTH:
                    self.images[face] = Depth(value)

        missing = self.REQUIRED_FACES - self.images.keys()
        if missing:
            raise ValueError(f"Missing faces: {', '.join(f.value for f in missing)}")

    @classmethod
    def load(cls, path: Path) -> "CubeMap":
        meta_path = path / cls.META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")
        
        meta = parse_json(meta_path.read_text())
        cube_type = CubeMapType(meta["type"])

        local_files = {}
        for subpath in path.iterdir():
            if subpath.is_file() and subpath.suffix.lower() == "." + cube_type.preferred_extension():
                local_files[subpath.stem] = subpath
        return cls(local_files, cube_type)
    
    def _file_name(self, root: Path, face: CubeFace) -> Path:
        return root / f"{face.value}.{self.type.preferred_extension()}"

    def save(self, path: Path):
        self._save(
            path=path,
            write_metadata=True
        )

    def _save(self, path: Path, write_metadata: bool):
        path.mkdir(
            parents=True, 
            exist_ok=True
        )

        if write_metadata:
            meta_path = path / self.META_FILE
            meta_path.write_text(write_json({"type": self.type.value}))

        for face, image in self.images.items():
            image.save(self._file_name(root=path, face=face)) 

    def map(self, fn: Callable) -> CubeMap:
        return CubeMap({face: fn(value) for face, value in self.images.items()}, self.type)

    def save_debug_image(self, path: Path):
        if self.type is not CubeMapType.DEPTH:
            self._save(
                path=path, 
                write_metadata=False
            )
            return

        path.mkdir(
            parents=True, 
            exist_ok=True
        )
        for face, image in self.images.items():
            image.save_debug_image(self._file_name(root=path, face=face)) 

    def copy(self) -> CubeMap:
        return self.map(lambda x: x.copy())
    
    def __getitem__(self, key: CubeFace) -> Image | Depth:
        return self.images[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CubeMap):
            return NotImplemented
        return self.type == other.type and self.images == other.images