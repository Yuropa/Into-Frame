import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from pipeline.pipeline_context import PipelineContext

EXTENSION = ".frame"


def create_frame_archive(
    context_dir: Path,
    input_path: Path,
    output_dir: Path,
    stage_order: list[str],
) -> Path:
    """Package a saved pipeline context directory into a .frame archive."""
    stem = Path(input_path).stem
    archive_path = output_dir / f"{stem}{EXTENSION}"

    manifest = {
        "input_name": Path(input_path).name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": stage_order,
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode()

    def _exclude_build(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        # Strip per-stage build/ directories — intermediate files, not needed for playback.
        parts = Path(tarinfo.name).parts
        if "build" in parts:
            return None
        return tarinfo

    with tarfile.open(archive_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))
        tar.add(context_dir, arcname="context", filter=_exclude_build)

    return archive_path


def load_frame_archive(archive_path: Path, extract_dir: Path) -> tuple[PipelineContext, list[str]]:
    """Extract a .frame archive and return a loaded PipelineContext and stage order."""
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    stages: list[str] = []
    manifest_path = extract_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        stages = manifest.get("stages", [])

    context = PipelineContext()
    context.load(extract_dir / "context", stages)
    return context, stages
