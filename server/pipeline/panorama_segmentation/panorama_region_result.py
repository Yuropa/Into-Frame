from dataclasses import dataclass, field
from enum import IntEnum
from typing import Self

import numpy as np


class RegionType(IntEnum):
    """
    Coarse semantic region types used across segmentation, height mapping, and
    region mapping.  Integer values are stable pixel-map indices (uint8 arrays
    store these directly).
    """
    SKY        = 0
    WATER      = 1
    TERRAIN    = 2
    GROUND     = 3
    VEGETATION = 4
    BUILT      = 5
    OTHER      = 6
    ROAD       = 7
    TRAIL      = 8

    @property
    def ground_valid(self) -> bool:
        """True for types whose depth can be trusted for ground-plane estimation.

        Vegetation canopies and built structures occlude the ground, so their
        projected depth cannot be used to build a reliable height or region map.
        """
        return self in (RegionType.WATER, RegionType.TERRAIN, RegionType.GROUND, RegionType.ROAD, RegionType.TRAIL)

    @property
    def label(self) -> str:
        """Lowercase string name, used in serialisation and display."""
        return self.name.lower()

    @classmethod
    def from_label(cls, s: str) -> "RegionType":
        return cls[s.upper()]


# Keyword substrings that map ADE20K label names to coarse region types.
# Checked in order; first match wins.
_LABEL_RULES: list[tuple[tuple[str, ...], RegionType]] = [
    (("sky",), RegionType.SKY),
    (("water", "sea", "ocean", "river", "lake", "pool", "waterfall", "fountain", "swamp"), RegionType.WATER),
    (("mountain", "hill", "cliff", "rock", "stone", "boulder", "land"), RegionType.TERRAIN),
    (("tree", "palm", "plant", "bush", "shrub", "flower", "vegetation", "forest", "jungle"), RegionType.VEGETATION),
    (("building", "house", "skyscraper", "hovel", "shed", "cabin", "tower", "church", "temple"), RegionType.BUILT),
    (("wall", "fence", "railing", "bannister", "column", "pillar"), RegionType.BUILT),
    (("road", "pavement", "runway"), RegionType.ROAD),
    (("sidewalk", "path", "trail"), RegionType.TRAIL),
    (("grass", "earth", "field", "sand", "dirt", "mud", "ground", "soil",
      "floor", "snow", "ice"), RegionType.GROUND),
]


def coarse_type_for_label(label_name: str) -> RegionType:
    name = label_name.lower()
    for keywords, region_type in _LABEL_RULES:
        if any(kw in name for kw in keywords):
            return region_type
    return RegionType.OTHER


REGION_TYPE_COLORS: dict[RegionType, tuple[int, int, int]] = {
    RegionType.SKY:        (135, 206, 235),
    RegionType.WATER:      (30,  144, 255),
    RegionType.TERRAIN:    (139, 90,  43),
    RegionType.GROUND:     (160, 120, 60),
    RegionType.VEGETATION: (34,  139, 34),
    RegionType.BUILT:      (169, 169, 169),
    RegionType.OTHER:      (200, 200, 200),
    RegionType.ROAD:       (80,  80,  80),
    RegionType.TRAIL:      (180, 140, 100),
}


def build_type_idx_map(label_map: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    """Map ADE20K per-pixel class IDs to a per-pixel coarse region type index array."""
    type_idx_map = np.full(label_map.shape, RegionType.OTHER, dtype=np.uint8)
    for class_id, label_name in id2label.items():
        region_type = coarse_type_for_label(label_name)
        type_idx_map[label_map == class_id] = int(region_type)
    return type_idx_map


def colorize_region_type_map(type_idx_map: np.ndarray) -> np.ndarray:
    """Convert a per-pixel region type index array (H, W) to an RGB image array (H, W, 3)."""
    rgb = np.zeros((*type_idx_map.shape, 3), dtype=np.uint8)
    for rt in RegionType:
        rgb[type_idx_map == rt] = REGION_TYPE_COLORS[rt]
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
