import dataclasses
from pathlib import Path
from util.json_utils import parse_json, write_json

class RemoteObject:
    def encode(self) -> str:
        return write_json(self.__dict__)

    @classmethod
    def decode(cls, json_str: str):
        return cls(**parse_json(json_str))

@dataclasses.dataclass
class Status(RemoteObject):
    status: str

    def __init__(self, status: str) -> None:    
        self.status = status

@dataclasses.dataclass
class RemoteInput(RemoteObject):
    action: str
    temp_path: str
    input: object

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