import sys
from pathlib import Path

def add_project_paths(start: Path = None):
    if start is None:
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

    lib_packages = server_path.parent / "lib" / "packages"

    # Add to sys.path if not already present
    for path in [server_path, lib_packages]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)