from pathlib import Path
from typing import Optional, Iterator
import hashlib
import uuid
from util.image_utils import Image
from util.path_utils import resource_directory

class PipelineInputItem:
    image: Image
    uuid: uuid

    def __init__(self, p: Path):
        self.image = Image(p)
        self.uuid = self._hash_image(p)

    def _hash_image(self, path: Path) -> uuid:
        with open(str(path), "rb") as f:
            digest = hashlib.md5(f.read()).digest()
            return str(uuid.UUID(bytes=digest))

    def uuid_string(self) -> str:
        return str(self.uuid)

    def equal_to(self, image: Image) -> bool:
        return self.image == image
    
    def path(self, root: Path) -> Path:
        return root / self.uuid_string()

class PipelineInput:
    path: Path

    def __init__(self, root: Optional[str | Path] = None):
        self.supported_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

        if root is None:
            self.path = resource_directory()
        elif isinstance(root, Path):
            self.path = root
        elif isinstance(root, str):
            if len(root) > 0:
                self.path = Path(root)
            else:
                self.path = resource_directory()
        else:
            raise RuntimeError(f"Unknown path type: {root!r}")

    def is_directory(self) -> bool:
        return self.path.is_dir()

    def _paths(self) -> Iterator[Path]: 
        return self.path.rglob("*") if self.is_directory() else iter([self.path])

    def count(self) -> int:
        return sum(1 for p in self._paths() if p.suffix.lower() in self.supported_extensions)

    def all_images(self) -> Iterator[PipelineInputItem]:
        for p in self._paths():
            if p.suffix.lower() in self.supported_extensions:
                yield PipelineInputItem(p)