from dataclasses import dataclass, field
from typing import Self

import numpy as np


REGION_TYPE_SKY = "sky"
REGION_TYPE_WATER = "water"
REGION_TYPE_TERRAIN = "terrain"
REGION_TYPE_GROUND = "ground"
REGION_TYPE_VEGETATION = "vegetation"
REGION_TYPE_BUILT = "built"
REGION_TYPE_OTHER = "other"

ALL_REGION_TYPES = (
    REGION_TYPE_SKY,
    REGION_TYPE_WATER,
    REGION_TYPE_TERRAIN,
    REGION_TYPE_GROUND,
    REGION_TYPE_VEGETATION,
    REGION_TYPE_BUILT,
    REGION_TYPE_OTHER,
)

# Keyword substrings that map ADE20K label names to coarse region types.
# Checked in order; first match wins.
_LABEL_RULES: list[tuple[tuple[str, ...], str]] = [
    (("sky",), REGION_TYPE_SKY),
    (("water", "sea", "ocean", "river", "lake", "pool", "waterfall", "fountain", "swamp"), REGION_TYPE_WATER),
    (("mountain", "hill", "cliff", "rock", "stone", "boulder", "land"), REGION_TYPE_TERRAIN),
    (("tree", "palm", "plant", "bush", "shrub", "flower", "vegetation", "forest", "jungle"), REGION_TYPE_VEGETATION),
    (("building", "house", "skyscraper", "hovel", "shed", "cabin", "tower", "church", "temple"), REGION_TYPE_BUILT),
    (("wall", "fence", "railing", "bannister", "column", "pillar"), REGION_TYPE_BUILT),
    (("grass", "earth", "field", "sand", "dirt", "mud", "ground", "soil",
      "path", "road", "sidewalk", "pavement", "runway", "floor", "snow", "ice"), REGION_TYPE_GROUND),
]


def coarse_type_for_label(label_name: str) -> str:
    name = label_name.lower()
    for keywords, region_type in _LABEL_RULES:
        if any(kw in name for kw in keywords):
            return region_type
    return REGION_TYPE_OTHER


REGION_TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    REGION_TYPE_SKY:        (135, 206, 235),
    REGION_TYPE_WATER:      (30,  144, 255),
    REGION_TYPE_TERRAIN:    (139, 90,  43),
    REGION_TYPE_GROUND:     (160, 120, 60),
    REGION_TYPE_VEGETATION: (34,  139, 34),
    REGION_TYPE_BUILT:      (169, 169, 169),
    REGION_TYPE_OTHER:      (200, 200, 200),
}


def build_type_idx_map(label_map: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    """Map ADE20K per-pixel class IDs to a per-pixel coarse region type index array."""
    other_idx = ALL_REGION_TYPES.index(REGION_TYPE_OTHER)
    type_idx_map = np.full(label_map.shape, other_idx, dtype=np.uint8)
    for class_id, label_name in id2label.items():
        region_type = coarse_type_for_label(label_name)
        type_idx = ALL_REGION_TYPES.index(region_type)
        type_idx_map[label_map == class_id] = type_idx
    return type_idx_map


def colorize_region_type_map(type_idx_map: np.ndarray) -> np.ndarray:
    """Convert a per-pixel region type index array (H, W) to an RGB image array (H, W, 3)."""
    rgb = np.zeros((*type_idx_map.shape, 3), dtype=np.uint8)
    for i, region_type in enumerate(ALL_REGION_TYPES):
        rgb[type_idx_map == i] = REGION_TYPE_COLORS.get(region_type, (200, 200, 200))
    return rgb


@dataclass
class PanoramaRegion:
    """One detected region in the panorama."""

    region_type: str
    label_name: str
    area_fraction: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in panorama pixels
    centroid: tuple[float, float]    # (cx, cy) in panorama pixels

    def encode(self) -> dict:
        return {
            "region_type": self.region_type,
            "label_name": self.label_name,
            "area_fraction": self.area_fraction,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        return cls(
            region_type=data["region_type"],
            label_name=data["label_name"],
            area_fraction=float(data["area_fraction"]),
            bbox=tuple(data["bbox"]),
            centroid=tuple(data["centroid"]),
        )


@dataclass
class PanoramaRegionResult:
    """
    Semantic region analysis of the equirectangular panorama.

    regions is a list of PanoramaRegion entries, one per connected component
    above the area threshold, sorted by area_fraction descending.
    present_types is the set of coarse region types actually found.
    type_fractions gives the total pixel fraction per coarse type.
    """

    regions: list[PanoramaRegion] = field(default_factory=list)
    present_types: list[str] = field(default_factory=list)
    type_fractions: dict[str, float] = field(default_factory=dict)

    def regions_of_type(self, region_type: str) -> list[PanoramaRegion]:
        return [r for r in self.regions if r.region_type == region_type]

    def dominant_type(self) -> str | None:
        if not self.type_fractions:
            return None
        return max(self.type_fractions, key=self.type_fractions.get)

    def encode(self) -> dict:
        return {
            "regions": [r.encode() for r in self.regions],
            "present_types": self.present_types,
            "type_fractions": self.type_fractions,
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        result = cls()
        result.regions = [PanoramaRegion.decode(r) for r in data.get("regions", [])]
        result.present_types = data.get("present_types", [])
        result.type_fractions = data.get("type_fractions", {})
        return result
