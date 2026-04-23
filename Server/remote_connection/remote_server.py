import sys
import torch
import socket
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from util.device_utils import clean_device_cache
from remote_connection.remote_types import RemoteInput, RemoteOutput, Status, RemoteObject

class RemoteServer(ABC):
    def __init__(self) -> None:
        name_device = sys.argv[1] if len(sys.argv) > 1 else "cpu"   
        self.device = torch.device(name_device) 

        sock_path = sys.argv[2]
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(sock_path)
        self.json_in = self.sock.makefile('r')
        self.json_out = self.sock.makefile('w')

    def close(self):
        self.json_in.close()
        self.json_out.close()
        self.sock.close()

    def connect(self):
        status = Status("ready")
        self._send(status)

    def _send(self, obj: RemoteObject):
        print(obj.encode(), file=self.json_out, flush=True)

    def poll(self):
        for line in self.json_in:
            request = RemoteInput.decode(line.strip())
            if request.action == "exit":
                break
            else:
                temp_path = Path(request.temp_path)
                try:
                    temp_path.mkdir(parents=True, exist_ok=True)
                    result_object = self.perform(
                        action=request.action, 
                        temp_path=temp_path, 
                        input=request.input
                    )
                    result = RemoteOutput(
                        action=request.action, 
                        output=result_object
                    )
                    self._send(result)
                except Exception as e:
                    result = RemoteOutput(action=request.action, output=None, error=str(e))
                    self._send(result)
                    return
                
                clean_device_cache(self.device)

    @abstractmethod
    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        return input

    @classmethod
    def run(cls):
        server = cls()
        server.connect()
        server.poll()
        server.close()