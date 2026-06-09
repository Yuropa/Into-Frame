import torch
import subprocess
from pathlib import Path
import threading
import socket
import tempfile
import shutil
import base64
import os
import time
import logging
from io import BytesIO

from typing import Any, Callable, Optional
from util.image_utils import Image
from util.device_utils import device_id
from util.json_utils import parse_json
from remote_connection.remote_types import RemoteInput, RemoteOutput, Status, RemoteObject

_log = logging.getLogger("pipeline.remote")

class RemoteClient():
    def _readline_json(self, pipe):
        while True:
            line = pipe.readline()
            if not line:
                return None  # EOF
            if line.strip():
                return line.strip()

    def _make_capture_thread(self, stream, lines, lock, prefix):
        def _capture(stream):
            for line in iter(stream.readline, b""):
                decoded = line.decode('utf-8', errors='replace')
                with lock:
                    lines.append(decoded)
                stripped = decoded.rstrip()
                if stripped:
                    _log.info("[%s] %s", prefix, stripped)
            stream.close()
        return threading.Thread(target=_capture, args=(stream,), daemon=True)

    def _cuda_env_for_device(self, device: torch.device, env_options: Optional[dict] = None) -> dict:
        env = os.environ.copy()

        if env_options:
            env.update({k: str(v) for k, v in env_options.items()})
        if device.type == "cuda":
            idx = device.index if device.index is not None else torch.cuda.current_device()
            env["CUDA_VISIBLE_DEVICES"] = str(idx)

        return env

    @staticmethod
    def _verify_conda_env(conda_env: str) -> None:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to query conda environments: {result.stderr}")
        envs = parse_json(result.stdout).get("envs", [])
        env_names = [Path(e).name for e in envs]
        if conda_env not in env_names:
            raise RuntimeError(f"Conda environment '{conda_env}' not found. Available: {env_names}")

    def __init__(self, device: torch.device, conda_env: str, script_path: Path, env_options: Optional[dict] = None) -> None:
        self._verify_conda_env(conda_env)
        self.process = None
        self.device = device
        self.script_path = script_path

        self.sock_dir = tempfile.mkdtemp()
        self.sock_path = str(Path(self.sock_dir) / "remote.sock")
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.sock_path)
        self.server_sock.listen(1)
        self.conda_env = conda_env

        env = self._cuda_env_for_device(device, env_options)

        self.process = subprocess.Popen(
            [
                "conda", "run", "--no-capture-output",
                "-n", conda_env,
                "python", "-u", str(script_path),
                device_id(device),
                self.sock_path
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=env
        )

        self._stdout_lines = []
        self._stdout_lock = threading.Lock()
        self._stderr_lines = []
        self._stderr_lock = threading.Lock()

        self._make_capture_thread(self.process.stdout, self._stdout_lines, self._stdout_lock, "out").start()
        self._make_capture_thread(self.process.stderr, self._stderr_lines, self._stderr_lock, "err").start()

        self.server_sock.settimeout(1)
        deadline = time.time() + 60
        while True:
            try:
                self.conn, _ = self.server_sock.accept()
                break
            except socket.timeout:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Subprocess exited (code {self.process.returncode}) before connecting "
                        f"(conda env '{conda_env}' may not exist):\n{self._get_stderr()}"
                    )
                if time.time() > deadline:
                    raise RuntimeError("Timed out waiting for subprocess to connect")

        self.json_pipe = self.conn.makefile('r')
        self.json_out = self.conn.makefile('w')

        ready_line = self._readline_json(self.json_pipe)
        if not ready_line:
            raise RuntimeError("subprocess produced no output")

        ready_status = Status.decode(ready_line)
        if ready_status.status != "ready":
            raise RuntimeError(f"Unexpected startup message: {ready_line}")

        _log.info("Loaded %s", script_path.name)

    def __del__(self):
        if self.process is not None:
            self.close()

    def _get_stdout(self) -> str:
        with self._stdout_lock:
            return "".join(self._stdout_lines)

    def _get_stderr(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_lines)

    def dump_logs(self):
        tag = f"[{self.conda_env}]"
        for line in self._get_stdout().splitlines():
            if line.strip():
                _log.debug("%s[stdout] %s", tag, line)
        for line in self._get_stderr().splitlines():
            if line.strip():
                _log.debug("%s[stderr] %s", tag, line)

    def _check_for_errors(self):
        if self.process.poll() is not None:
            self.dump_logs()
            raise RuntimeError(f"subprocess exited before running:\n{self._get_stderr()}")

    def _send(self, obj: RemoteObject):
        encoded = obj.encode()
        self.json_out.write(encoded)
        self.json_out.flush()

    def send(
        self,
        action: str,
        input,
        temp_path: Path,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> Any:
        self._check_for_errors()

        request = RemoteInput(
            action=action,
            temp_path=temp_path,
            input=input
        )
        self._send(request)
        self.dump_logs()

        while True:
            response_line = self._readline_json(self.json_pipe)
            if response_line is None:
                raise RuntimeError("Subprocess closed connection unexpectedly")

            raw = parse_json(response_line)
            if isinstance(raw, dict) and raw.get("type") == "progress":
                if on_progress is not None:
                    on_progress(float(raw.get("fraction", 0.0)), str(raw.get("label", "")))
                continue

            response = RemoteOutput.decode(response_line)
            error = getattr(response, "error", None)
            if error:
                stack = getattr(response, "stack", None)
                _log.error("Remote error: %s\n%s", error, stack or "")
                raise RuntimeError(f"Remote error: {error}")

            self.dump_logs()
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