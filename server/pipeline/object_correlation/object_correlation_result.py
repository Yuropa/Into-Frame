from dataclasses import dataclass, field


@dataclass
class ObjectGroupStats:
    """Per-type statistics for a group of correlated objects."""

    object_type: str
    indices: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.indices)


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
