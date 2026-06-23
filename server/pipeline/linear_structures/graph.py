import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import KDTree


@dataclass
class LinearStructure:
    """A single road, river, trail, etc. as a polyline in world space."""
    type: str          # "road" | "river" | "trail"
    path: np.ndarray   # (K, 3) float32 — world (x, y, z) along the centreline
    width: float       # estimated width in metres


class LinearGraph:
    """
    Spatial index of all detected linear structures.

    Queries are answered in XZ (the horizontal plane) only — Y is ignored
    since terrain height varies along a structure.
    """

    def __init__(self) -> None:
        self.structures: list[LinearStructure] = []
        self._tree: KDTree | None = None
        self._pts_xz: np.ndarray | None = None
        self._pts_types: list[str] = []

    def add(self, structure: LinearStructure) -> None:
        self.structures.append(structure)
        self._tree = None  # invalidate index

    def _ensure_index(self) -> None:
        if self._tree is not None or not self.structures:
            return
        xz, types = [], []
        for s in self.structures:
            xz.append(s.path[:, [0, 2]])
            types.extend([s.type] * len(s.path))
        self._pts_xz = np.concatenate(xz, axis=0).astype(np.float32)
        self._pts_types = types
        self._tree = KDTree(self._pts_xz)

    def nearest(
        self,
        x: float,
        z: float,
        types: set[str] | None = None,
    ) -> tuple[float, str]:
        """Return (distance_m, type) of the nearest matching structure."""
        self._ensure_index()
        if self._tree is None:
            return float("inf"), ""
        if types is None:
            d, i = self._tree.query([x, z])
            return float(d), self._pts_types[int(i)]
        # Filter to requested types
        mask = np.array([t in types for t in self._pts_types], dtype=bool)
        if not mask.any():
            return float("inf"), ""
        pts = self._pts_xz[mask]
        idx_map = np.where(mask)[0]
        d, i = KDTree(pts).query([x, z])
        return float(d), self._pts_types[idx_map[int(i)]]

    def is_near(
        self,
        x: float,
        z: float,
        max_dist: float,
        types: set[str] | None = None,
    ) -> bool:
        d, _ = self.nearest(x, z, types)
        return d <= max_dist

    def by_type(self) -> dict[str, list[LinearStructure]]:
        result: dict[str, list[LinearStructure]] = {}
        for s in self.structures:
            result.setdefault(s.type, []).append(s)
        return result

    def __len__(self) -> int:
        return len(self.structures)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for s in self.structures:
            counts[s.type] = counts.get(s.type, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "empty"
