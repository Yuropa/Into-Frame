import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image as PILImage
from pipeline.pipeline_context import PipelineContext

EXTENSION = ".frame"


def _add_dir(tar: tarfile.TarFile, src_dir: Path, arcname: str, filter_fn) -> None:
    """Walk src_dir into tar, stripping PNG internal compression so xz compresses pixels directly."""
    tar.addfile(filter_fn(tar.gettarinfo(str(src_dir), arcname=arcname)))
    for path in sorted(src_dir.rglob("*")):
        name = f"{arcname}/{path.relative_to(src_dir)}"
        tarinfo = filter_fn(tar.gettarinfo(str(path), arcname=name))
        if tarinfo is None:
            continue
        if path.is_dir():
            tar.addfile(tarinfo)
        elif path.suffix.lower() == ".png":
            # Store pixels uncompressed so xz can exploit cross-image redundancy.
            buf = io.BytesIO()
            PILImage.open(path).save(buf, format="PNG", compress_level=0)
            data = buf.getvalue()
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        else:
            with open(path, "rb") as f:
                tar.addfile(tarinfo, f)


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

    with tarfile.open(archive_path, "w:xz", preset=9) as tar:
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))
        _add_dir(tar, context_dir, arcname="context", filter_fn=_exclude_build)

    return archive_path


def load_frame_archive(archive_path: Path, extract_dir: Path) -> tuple[PipelineContext, list[str]]:
    """Extract a .frame archive and return a loaded PipelineContext and stage order."""
    with tarfile.open(archive_path, "r:*") as tar:
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
