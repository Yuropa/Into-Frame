from dataclasses import dataclass, field
from typing import Self


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
