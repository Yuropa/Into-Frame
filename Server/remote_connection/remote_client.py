import torch
import subprocess
from pathlib import Path
import threading
import socket
import tempfile
import shutil
import base64
from io import BytesIO

from typing import Any
from util.image_utils import Image
from util.device_utils import device_id
from remote_connection.remote_types import RemoteInput, RemoteOutput, Status, RemoteObject

class RemoteClient():
    def _readline_json(self, pipe):
        while True:
            line = pipe.readline()
            if not line:
                return None  # EOF
            if line.strip():
                return line.strip()

    def _pipe_stream(self, stream, prefix = None):
        if prefix is None:
            prefix = ""
        else:
            prefix = f"[{prefix}]"
        
        for line in iter(stream.readline, b""):
            print(f"{prefix} {line.decode('utf-8', errors='replace')}", end="", flush=True)

        stream.close()
    
    def __init__(self, device: torch.device, conda_env: str, script_path: Path) -> None:
        self.process = None
        self.device = device
        self.script_path = script_path

        self.sock_dir = tempfile.mkdtemp()
        self.sock_path = str(Path(self.sock_dir) / "remote.sock")
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.sock_path)
        self.server_sock.listen(1)

        self.process = subprocess.Popen(
            ["conda", "run", "--no-capture-output", "-n", conda_env, "python", str(script_path), device_id(device), self.sock_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=False
        )

        threading.Thread(target=self._pipe_stream, args=(self.process.stderr, "err"), daemon=True).start()
        threading.Thread(target=self._pipe_stream, args=(self.process.stdout, "out"), daemon=True).start()

        self.server_sock.settimeout(60)
        try:
            self.conn, _ = self.server_sock.accept()
        except socket.timeout:
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                raise RuntimeError(f"Subprocess died while waiting:\n{stderr}")
            raise RuntimeError("Timed out waiting for subprocess to connect")

        self.json_pipe = self.conn.makefile('r')
        self.json_out = self.conn.makefile('w')

        ready_line = self._readline_json(self.json_pipe)
        if not ready_line:
            raise RuntimeError("subprocess produced no output")

        ready_status = Status.decode(ready_line)
        if ready_status.status != "ready":
            raise RuntimeError(f"Unexpected startup message: {ready_line}")

        print(f"Finished loading script {script_path}")

    def __del__(self):
        if self.process is not None:
            self.close()
    
    def _check_for_errors(self):
        if self.process.poll() is not None:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"subprocess exited before running:\n{stderr}")


    def _send(self, obj: RemoteObject):
        encoded = obj.encode()
        self.json_out.write(encoded)
        self.json_out.flush()

    def send(self, action: str, input, temp_path: Path) -> Any:
        self._check_for_errors()

        request = RemoteInput(
            action=action, 
            temp_path=temp_path, 
            input=input
        )
        self._send(request)

        response_line = self._readline_json(self.json_pipe)
        if response_line is None:
            raise RuntimeError("Subprocess closed connection unexpectedly")
        response = RemoteOutput.decode(response_line)

        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"Remote error: {error}")

        return response.output
    
    def close(self):
        if self.process is not None:
            try:
                if hasattr(self, 'json_out'):
                    request = Status("exit")
                    self._send(request)
            except (BrokenPipeError, OSError):
                pass
            self.process.wait()
            if hasattr(self, 'server_sock'):
                self.server_sock.close()
                del self.server_sock
            if hasattr(self, 'sock_path'):
                Path(self.sock_path).unlink(missing_ok=True)
            if hasattr(self, 'conn'):
                self.conn.close()
                del self.conn
            if hasattr(self, 'sock_dir'):
                shutil.rmtree(self.sock_dir, ignore_errors=True)
            self.process = None

    def encode_image(self, image: Image):
        buffer = BytesIO()
        image.image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()