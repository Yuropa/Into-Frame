from pathlib import Path

def resource_directory() -> Path:
    BASE = Path(__file__).parent
    return BASE / ".." / "samples"
