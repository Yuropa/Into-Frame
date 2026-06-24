from dataclasses import dataclass, field


@dataclass
class ReportSection:
    """A single stage's contribution to the pipeline report."""
    stage_name: str
    title: str
    body: str = ""
    images: list = field(default_factory=list)  # [(PIL.Image, caption_str), ...]
    stats: dict = field(default_factory=dict)   # {label: value_str, ...} (ordered)
