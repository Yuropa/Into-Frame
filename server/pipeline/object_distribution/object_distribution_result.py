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

    def encode(self) -> dict:
        return {
            "object_type": self.object_type,
            "region_type": self.region_type,
            "n_points": self.n_points,
            "bin_count": self.bin_count,
            "hist": self.hist,
            "pair_count": self.pair_count,
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
        return obj


@dataclass
class ObjectDistributionResult:
    """
    Per-region, per-type Voronoi PCF histograms produced by ObjectDistributionStage.

    distributions[region_type][object_type] gives the TypeDistribution for that
    combination. Combinations with fewer than two instances are absent. When no
    panorama region data is available all objects are grouped under "global".
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
