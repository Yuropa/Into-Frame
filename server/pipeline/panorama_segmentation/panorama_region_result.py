from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple, Optional, Self

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


# A rule's ambiguity resolution strategy -- how downstream code decides
# whether to trust a region matching this rule, when its coarse type isn't
# RegionType.ground_valid (see PanoramaRegion.well_supported / _build_result):
#
#   None            -- never questioned. The default for every rule that
#                       isn't routinely confused with something else.
#   "corroboration" -- trusted only if a *different*, unambiguous region of
#                       the same coarse type is independently present
#                       elsewhere in the panorama. Safe when real instances
#                       of this coarse type are rare in outdoor photography
#                       (e.g. BUILT structures) -- their total absence
#                       elsewhere is itself strong evidence. Unsafe for a
#                       coarse type that's ubiquitous and unremarkable in
#                       nature photos: two separate real-looking regions
#                       don't corroborate each other if the *whole coarse
#                       type* is prone to the same confusion (this is why
#                       VEGETATION doesn't use this strategy -- see
#                       "confidence" below).
#   "confidence"    -- trusted unless BOTH this region's own mean
#                       top1-vs-runner-up softmax margin is low (a
#                       genuinely close call for the model, not a
#                       comfortable one) AND its dominant runner-up type is
#                       itself ground-valid (there's a plausible resolution
#                       to fall back to). Used where corroboration-by-
#                       absence is unsafe because real instances of this
#                       rule's coarse type are common in outdoor photography
#                       (e.g. VEGETATION -- almost every nature photo has
#                       some legitimate bush or tree, so "no other
#                       vegetation elsewhere" says nothing about any one
#                       region), but a confused, low-margin call against a
#                       plausible ground-valid alternative still does.
AMBIGUITY_CORROBORATION = "corroboration"
AMBIGUITY_CONFIDENCE = "confidence"


class _LabelRule(NamedTuple):
    keywords: tuple[str, ...]
    region_type: RegionType
    ambiguity: Optional[str] = None


# Keyword substrings that map ADE20K label names to coarse region types.
# Checked in order; first match wins.
_LABEL_RULES: list[_LabelRule] = [
    _LabelRule(("sky",), RegionType.SKY),
    _LabelRule(("water", "sea", "ocean", "river", "lake", "pool", "waterfall", "fountain", "swamp"), RegionType.WATER),
    _LabelRule(("mountain", "hill", "cliff", "rock", "stone", "boulder", "land"), RegionType.TERRAIN),
    # Commonly confused with dark/mottled natural rock/cliff faces (a dense,
    # high-frequency, clump-like texture, especially in shadow) by
    # ADE20K-trained segmentation models. Unlike "wall" below, real instances
    # of this coarse type are common and unremarkable in outdoor photography,
    # so "confidence" (not "corroboration") is the safe strategy here -- see
    # the strategy docstring above.
    _LabelRule(("tree", "palm", "plant", "bush", "shrub", "flower", "vegetation", "forest", "jungle"), RegionType.VEGETATION, ambiguity=AMBIGUITY_CONFIDENCE),
    _LabelRule(("building", "house", "skyscraper", "hovel", "shed", "cabin", "tower", "church", "temple"), RegionType.BUILT),
    # Commonly confused with sunlit natural rock/cliff faces (large, evenly-
    # lit, texture-rich vertical surfaces) by ADE20K-trained segmentation
    # models, whose "wall" class is dominated by indoor/urban training data.
    # Real BUILT structures are rare and remarkable in outdoor/nature
    # photography, so "corroboration" (not "confidence") is the safe
    # strategy here -- see the strategy docstring above.
    _LabelRule(("wall", "fence", "railing", "bannister", "column", "pillar"), RegionType.BUILT, ambiguity=AMBIGUITY_CORROBORATION),
    _LabelRule(("road", "pavement", "runway"), RegionType.ROAD),
    _LabelRule(("sidewalk", "path", "trail"), RegionType.TRAIL),
    _LabelRule(("grass", "earth", "field", "sand", "dirt", "mud", "ground", "soil",
      "floor", "snow", "ice"), RegionType.GROUND),
]


def coarse_type_for_label(label_name: str) -> RegionType:
    name = label_name.lower()
    for rule in _LABEL_RULES:
        if any(kw in name for kw in rule.keywords):
            return rule.region_type
    return RegionType.OTHER


def ambiguity_strategy_for_label(label_name: str) -> Optional[str]:
    """The matching rule's ambiguity resolution strategy (see _LabelRule /
    AMBIGUITY_CORROBORATION / AMBIGUITY_CONFIDENCE), or None if this label
    is never questioned."""
    name = label_name.lower()
    for rule in _LABEL_RULES:
        if any(kw in name for kw in rule.keywords):
            return rule.ambiguity
    return None


# Region types treated as one interchangeable "ground-like" domain for object
# distribution synthesis: ADE20K segmentation routinely splits a single walkable
# area into these three labels (grass underfoot vs. a distant hillside vs. a
# tree's own canopy shadow), even though it's all one contiguous surface a
# population of objects should scatter across. Everything else (WATER, BUILT,
# ROAD, TRAIL, SKY, OTHER) is left as an exact self-match only.
_GROUND_PAINT_GROUP: frozenset[RegionType] = frozenset({
    RegionType.TERRAIN, RegionType.GROUND, RegionType.VEGETATION,
})


def paintable_region_types(region_type: RegionType) -> frozenset[RegionType]:
    """Region types eligible to receive a distribution learned from exemplars
    observed on `region_type` (see DistributionSynthesisStage). Returns the
    shared ground-like group for TERRAIN/GROUND/VEGETATION, or just
    `{region_type}` for anything else.
    """
    if region_type in _GROUND_PAINT_GROUP:
        return _GROUND_PAINT_GROUP
    return frozenset({region_type})


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
    # Mean top-1 softmax confidence of the segmentation model over this
    # region's pixels. 1.0 (neutral/unknown) when confidence wasn't computed
    # (e.g. decoded from an older saved result).
    mean_confidence: float = 1.0
    # Mean top1-vs-runner-up softmax margin over this region's pixels (see
    # AMBIGUITY_CONFIDENCE). 1.0 (neutral/unknown, maximally confident) when
    # not computed.
    mean_margin: float = 1.0
    # The ambiguity_strategy_for_label result for this region's label_name --
    # None if this label is never questioned, otherwise which mechanism
    # decided well_supported below (see _LabelRule / _build_result).
    ambiguity_strategy: Optional[str] = None
    # Only meaningful when ambiguity_strategy is not None: whether this
    # region's classification held up under that strategy. False means it
    # didn't, and every pixel in it has been resolved to its own runner-up
    # coarse type in PANORAMA_REGION_TYPE_MAP (see PanoramaRegionStage._build_result).
    well_supported: bool = True

    def encode(self) -> dict:
        return {
            "region_type": self.region_type,
            "label_name": self.label_name,
            "area_fraction": self.area_fraction,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "mean_confidence": self.mean_confidence,
            "mean_margin": self.mean_margin,
            "ambiguity_strategy": self.ambiguity_strategy,
            "well_supported": self.well_supported,
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        return cls(
            region_type=data["region_type"],
            label_name=data["label_name"],
            area_fraction=float(data["area_fraction"]),
            bbox=tuple(data["bbox"]),
            centroid=tuple(data["centroid"]),
            mean_confidence=float(data.get("mean_confidence", 1.0)),
            mean_margin=float(data.get("mean_margin", 1.0)),
            ambiguity_strategy=data.get("ambiguity_strategy"),
            well_supported=bool(data.get("well_supported", True)),
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
