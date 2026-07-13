import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink src to dst, falling back to a real copy if that's not possible.

    Use this instead of shutil.copy* whenever src is an already-encoded file on
    disk (video, PDF, ...) rather than an in-memory object being serialized —
    a hardlink makes both paths valid for the same bytes at zero extra disk
    cost when they share a filesystem. Falls back to a full copy across
    filesystem boundaries (os.link raises OSError, e.g. errno EXDEV) or on
    filesystems without hardlink support.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
