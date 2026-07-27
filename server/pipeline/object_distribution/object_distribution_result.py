from dataclasses import dataclass, field
from typing import Self


@dataclass
class TypeDistribution:
    """PCF hop-count histogram for a single object type within a single region type."""

    object_type: str
    region_type: str
    n_points: int
    bin_count: int
    hist: list[int] = field(default_factory=list)
    pair_count: int = 0
    # Raw exemplar data backing this distribution, in world-space XZ meters — the
    # points DistributionSynthesisStage feeds to synthesize_pattern as the exemplar
    # set, and the (width, height) footprints it samples from for synthesized points.
    # synthesize_pattern re-derives its own PCF from these directly; `hist` above is
    # for inspection/reporting only.
    points: list[tuple[float, float]] = field(default_factory=list)
    sizes: list[tuple[float, float]] = field(default_factory=list)
    # ObjectCategoryClusteringStage's visual variant id for each exemplar, index-
    # aligned with points/sizes. The spatial pattern is modelled per CLASS, not per
    # bucket -- a bucket is a per-instance appearance attribute, and splitting the
    # point set by it would fragment most groups below the two-point minimum a PCF
    # needs. Instead DistributionSynthesisStage draws a whole exemplar (its size AND
    # its bucket together) for each painted point, so the painted population inherits
    # the observed mix of variants and each instance keeps a self-consistent
    # size/appearance pairing. Without this the clustering result never reached
    # synthesis at all and every painted instance fell back to bucket 0 in
    # SceneGenerationStage -- one variant repeated across the whole region.
    buckets: list[int] = field(default_factory=list)

    def encode(self) -> dict:
        return {
            "object_type": self.object_type,
            "region_type": self.region_type,
            "n_points": self.n_points,
            "bin_count": self.bin_count,
            "hist": self.hist,
            "pair_count": self.pair_count,
            "points": [list(p) for p in self.points],
            "sizes": [list(s) for s in self.sizes],
            "buckets": list(self.buckets),
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        obj = cls(
            object_type=data["object_type"],
            region_type=data.get("region_type", "global"),
            n_points=data["n_points"],
            bin_count=data["bin_count"],
            pair_count=data.get("pair_count", 0),
        )
        obj.hist = data.get("hist", [])
        obj.points = [tuple(p) for p in data.get("points", [])]
        obj.sizes = [tuple(s) for s in data.get("sizes", [])]
        # Older caches predate buckets; fall back to variant 0 for every exemplar so
        # a stale OBJECT_DISTRIBUTION still decodes and paints (as it did before).
        obj.buckets = [int(b) for b in data.get("buckets", [])] or [0] * len(obj.points)
        return obj


@dataclass
class ObjectDistributionResult:
    """
    Per-region, per-type Voronoi PCF histograms produced by ObjectDistributionStage.

    distributions[region_type][object_type] gives the TypeDistribution for that
    combination, where region_type is a top-down REGION_MAP label (e.g.
    "vegetation", "ground"). Combinations with fewer than two instances are absent.
    """

    distributions: dict[str, dict[str, TypeDistribution]] = field(default_factory=dict)

    def distribution_for(self, object_type: str, region_type: str) -> TypeDistribution | None:
        return self.distributions.get(region_type, {}).get(object_type)

    def region_types(self) -> list[str]:
        return list(self.distributions.keys())

    def object_types_for_region(self, region_type: str) -> list[str]:
        return list(self.distributions.get(region_type, {}).keys())

    def encode(self) -> dict:
        return {
            "distributions": {
                region: {obj_type: dist.encode() for obj_type, dist in by_type.items()}
                for region, by_type in self.distributions.items()
            },
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        result = cls()
        for region, by_type in data.get("distributions", {}).items():
            result.distributions[region] = {
                obj_type: TypeDistribution.decode(dist)
                for obj_type, dist in by_type.items()
            }
        return result
