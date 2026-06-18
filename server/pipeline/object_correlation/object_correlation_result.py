from dataclasses import dataclass, field
from typing import Self


@dataclass
class ObjectGroupStats:
    """Per-type statistics for a group of correlated objects."""

    object_type: str
    indices: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.indices)

    def encode(self) -> dict:
        return {"object_type": self.object_type, "indices": self.indices}

    @classmethod
    def decode(cls, data: dict) -> Self:
        obj = cls(object_type=data["object_type"])
        obj.indices = data["indices"]
        return obj


@dataclass
class ObjectCorrelationResult:
    """
    Aggregated correlation data produced by ObjectCorrelationStage.

    groups maps each object type label to an ObjectGroupStats instance holding
    the indices of all surviving metadata_{i} entries that share that type.
    deduplicated_count is the number of Grounding DINO detections dropped because
    they overlapped an existing SAM2 detection.
    """

    groups: dict[str, ObjectGroupStats] = field(default_factory=dict)
    deduplicated_count: int = 0

    def group_for(self, object_type: str) -> ObjectGroupStats | None:
        return self.groups.get(object_type)

    def types(self) -> list[str]:
        return list(self.groups.keys())

    def encode(self) -> dict:
        return {
            "groups": {k: v.encode() for k, v in self.groups.items()},
            "deduplicated_count": self.deduplicated_count,
        }

    @classmethod
    def decode(cls, data: dict) -> Self:
        result = cls(deduplicated_count=data.get("deduplicated_count", 0))
        result.groups = {k: ObjectGroupStats.decode(v) for k, v in data.get("groups", {}).items()}
        return result
