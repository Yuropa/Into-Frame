from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional, Self


class Video:
    """A generated video file (MP4, optionally with a muxed audio track) on disk.

    Unlike Image/Depth/Panorama, video is never held in memory as decoded frames —
    this just wraps a path to the encoded file, the same on-disk convention Mesh
    uses for GLBs.
    """

    def __init__(self, obj, fps: Optional[float] = None, num_frames: Optional[int] = None):
        if isinstance(obj, str):
            obj = Path(obj)

        if isinstance(obj, Video):
            self.path = obj.path
            self.fps = obj.fps if fps is None else fps
            self.num_frames = obj.num_frames if num_frames is None else num_frames
        elif isinstance(obj, Path):
            self.path = obj
            self.fps = fps
            self.num_frames = num_frames
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(path)

    def save(self, path):
        path = Path(path)
        if path.resolve() != self.path.resolve():
            shutil.copyfile(self.path, path)

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size
