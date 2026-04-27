import base64
import dataclasses
import io
from pathlib import Path
import numpy as np
from util.json_utils import parse_json, write_json

def encode_value(v):
    if dataclasses.is_dataclass(v):
        return {k: encode_value(val) for k, val in dataclasses.asdict(v).items()}
    if isinstance(v, dict):
        return {k: encode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [encode_value(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (bytes, bytearray)):
        return {
            "__bytes__": True,
            "base64": base64.b64encode(v).decode("ascii")
        }
    if isinstance(v, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, v)
        return {
            "__ndarray__": True,
            "base64": base64.b64encode(buf.getvalue()).decode("ascii")
        }
    return None

def decode_value(v):
    if isinstance(v, dict):
        if v.get("__bytes__") is True:
            return base64.b64decode(v["base64"])
        if v.get("__ndarray__") is True:
            buf = io.BytesIO(base64.b64decode(v["base64"]))
            return np.load(buf)
        return {k: decode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [decode_value(x) for x in v]
    return v

class RemoteObject:
    def encode(self) -> str:
        payload = encode_value(self)
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

    def __init__(self, action: str, output, error = "") -> None:
        self.action = action
        self.output = output
        self.error = error