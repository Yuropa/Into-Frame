import sys
from pathlib import Path

def _server_path() -> Path:
    start = Path(__file__).resolve()
    current = start.parent
    server_path = None

    # Walk upward until we find "Server"
    for parent in [current] + list(current.parents):
        if parent.name == "Server":
            server_path = parent
            break

    if server_path is None:
        raise RuntimeError("Could not find 'Server' directory in path")

    return server_path

def add_system_path(path: Path):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

def add_project_paths():
    server_path = _server_path()

    lib_packages = server_path.parent / "lib" / "packages"

    # Add to sys.path if not already present
    for path in [server_path, lib_packages]:
        add_system_path(path)

def checkpoints_path() -> Path:
    return _server_path().parent / "checkpoints"

def lib_path() -> Path:
    return _server_path().parent / "lib"