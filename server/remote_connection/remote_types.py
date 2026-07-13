from __future__ import annotations

import base64
import dataclasses
import io
import uuid
from pathlib import Path
from typing import Optional
import numpy as np
from util.json_utils import parse_json, write_json
from util.image_utils import Image
from PIL import Image as PILImage

# Client and server for a remote stage always share a filesystem (the subprocess
# is spawned locally and temp_path is a directory both sides already read/write),
# so encode_value spills anything image/array-sized to a real file there and
# embeds just a path reference instead of paying for a base64 copy embedded in
# the JSON envelope. temp_path is None only for payloads that are never going to
# carry a heavy binary value (e.g. Status), so those keep the inline base64 path.

def _encode_ndarray_inline(v: np.ndarray) -> dict:
    buf = io.BytesIO()
    np.save(buf, v)
    return {"__ndarray__": True, "base64": base64.b64encode(buf.getvalue()).decode("ascii")}

def _encode_pil_image_inline(v: PILImage.Image, fmt: str) -> dict:
    buf = io.BytesIO()
    v.save(buf, format=fmt)
    return {"__pil_image__": True, "format": fmt, "base64": base64.b64encode(buf.getvalue()).decode("ascii")}

def encode_value(v, k = None, temp_path: Optional[Path] = None):
    if dataclasses.is_dataclass(v):
        # Field-by-field with getattr rather than dataclasses.asdict(v), which
        # deep-copies every leaf value (including any embedded image/ndarray)
        # before we'd get a chance to spill it to disk instead.
        return {f.name: encode_value(getattr(v, f.name), f.name, temp_path) for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {key: encode_value(val, key, temp_path) for key, val in v.items()}
    if isinstance(v, list):
        return [encode_value(x, k, temp_path) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (bytes, bytearray)):
        return {
            "__bytes__": True,
            "base64": base64.b64encode(v).decode("ascii")
        }
    if isinstance(v, np.ndarray):
        if temp_path is None:
            return _encode_ndarray_inline(v)
        file_path = temp_path / f"{uuid.uuid4().hex}.npy"
        np.save(file_path, v)
        return {"__ndarray_file__": True, "path": str(file_path)}
    if isinstance(v, Image):
        return encode_value(v.image, k, temp_path)
    if isinstance(v, PILImage.Image):
        fmt = v.format or "PNG"
        if temp_path is None:
            return _encode_pil_image_inline(v, fmt)
        file_path = temp_path / f"{uuid.uuid4().hex}.{fmt.lower()}"
        v.save(file_path, format=fmt)
        return {"__pil_image_file__": True, "format": fmt, "path": str(file_path)}
    print(f"Unable to encode value {v} for key {k}")
    return None

def decode_value(v):
    if isinstance(v, dict):
        if v.get("__bytes__") is True:
            return base64.b64decode(v["base64"])
        if v.get("__ndarray__") is True:
            buf = io.BytesIO(base64.b64decode(v["base64"]))
            return np.load(buf)
        if v.get("__ndarray_file__") is True:
            path = Path(v["path"])
            try:
                return np.load(path)
            finally:
                path.unlink(missing_ok=True)
        if v.get("__pil_image__") is True:
            buf = io.BytesIO(base64.b64decode(v["base64"]))
            return PILImage.open(buf).copy()
        if v.get("__pil_image_file__") is True:
            path = Path(v["path"])
            try:
                image = PILImage.open(path)
                image.load()  # force the decode now — the file is deleted right after
                return image
            finally:
                path.unlink(missing_ok=True)
        return {k: decode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [decode_value(x) for x in v]
    return v

class RemoteObject:
    def encode(self, temp_path: Optional[Path] = None) -> str:
        payload = encode_value(self, temp_path=temp_path)
        return write_json(payload)

    @classmethod
    def decode(cls, json_str: str):
        data = decode_value(parse_json(json_str))
        return cls(**data)

@dataclasses.dataclass
class Status(RemoteObject):
    status: str

    def __init__(self, status: str) -> None:    
        self.status = status

@dataclasses.dataclass
class RemoteInput(RemoteObject):
    action: str
    temp_path: str
    input: dict | list | str | int | float | bool | None

    def __init__(self, action: str, temp_path: Path, input) -> None:
        self.action = action
        self.temp_path = str(temp_path)
        self.input = input

@dataclasses.dataclass
class RemoteOutput(RemoteObject):
    action: str
    output: object
    error: object
    stack: object

    def __init__(self, action: str, output, error = "", stack="") -> None:
        self.action = action
        self.output = output
        self.error = error
        self.stack = stack