import sys
import torch
import socket
import base64
import traceback
from io import BytesIO
from PIL import Image
from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path
from util.device_utils import clean_device_cache, device_name, device_from_id
from util.json_utils import write_json
from remote_connection.remote_types import RemoteInput, RemoteOutput, Status, RemoteObject

class RemoteServer(ABC):
    def __init__(self) -> None:
        name_device = sys.argv[1] if len(sys.argv) > 1 else "cpu"   
        self.device = device_from_id(name_device) 

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

        print(f"Using device {device_name(self.device)}")

    def setup(self):
        pass

    def _send(self, obj: RemoteObject, temp_path: Optional[Path] = None):
        encoded = obj.encode(temp_path=temp_path)
        print(encoded, file=self.json_out, flush=True)

    def report_progress(self, fraction: float, label: str = ""):
        """Send an intermediate progress update to the client before the final result."""
        msg = write_json({"type": "progress", "fraction": fraction, "label": label})
        print(msg, file=self.json_out, flush=True)

    def poll(self):
        is_running = True
        for line in self.json_in:
            try:
                request = RemoteInput.decode(line.strip())
            except Exception as e:
                try:
                    status = Status.decode(line.strip())
                    if status.status == "exit":
                        is_running = False  
                        break
                    else:
                        continue
                except Exception as e:
                    print(f"Failed to decode request, skipping: {e}")
                    is_running = False
                    break

            if request.action == "exit":
                is_running = False
                break
            else:
                temp_path = Path(request.temp_path)
                try:
                    temp_path.mkdir(parents=True, exist_ok=True)
                    print(f"dispatch: {request.action}", flush=True)
                    result_object = self.perform(
                        action=request.action, 
                        temp_path=temp_path, 
                        input=request.input
                    )
                    result = RemoteOutput(
                        action=request.action,
                        output=result_object
                    )
                    self._send(result, temp_path=temp_path)
                except Exception as e:
                    stack_text = traceback.format_exc()
                    result = RemoteOutput(action=request.action, output=None, error=str(e), stack=stack_text)
                    self._send(result, temp_path=temp_path)
                    return
                
                clean_device_cache(self.device)
            if is_running is False:
                break

    @abstractmethod
    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        return input

    def decode_image(self, input: Any) -> Image:
        image_data = base64.b64decode(input)
        return Image.open(BytesIO(image_data))

    @classmethod
    def run(cls):
        server = cls()
        server.setup()
        server.connect()
        server.poll()
        server.close()