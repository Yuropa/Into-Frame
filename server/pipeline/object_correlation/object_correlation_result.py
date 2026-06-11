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
    the indices of all metadata_{i} entries that share that type.
    """

    groups: dict[str, ObjectGroupStats] = field(default_factory=dict)

    def group_for(self, object_type: str) -> ObjectGroupStats | None:
        return self.groups.get(object_type)

    def types(self) -> list[str]:
        return list(self.groups.keys())
