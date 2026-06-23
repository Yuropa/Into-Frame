import numpy as np
from scipy.spatial import KDTree

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey


# Keyword sets for broad semantic categories, checked in priority order.
_CATEGORY_KEYWORDS: list[tuple[str, frozenset[str]]] = [
    ("person",  frozenset({"person", "people", "man", "woman", "child", "boy", "girl", "human", "crowd", "figure", "pedestrian"})),
    ("vehicle", frozenset({"car", "vehicle", "truck", "bus", "bicycle", "motorcycle", "bike", "scooter", "van", "taxi", "automobile"})),
    ("animal",  frozenset({"animal", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "deer", "fox", "rabbit"})),
    ("tree",    frozenset({"tree", "trees", "palm", "pine", "oak", "birch", "maple", "trunk", "sapling"})),
    ("bush",    frozenset({"bush", "shrub", "hedge", "shrubbery", "plant", "plants", "vegetation", "foliage"})),
    ("bench",   frozenset({"bench", "seat", "seating"})),
    ("sign",    frozenset({"sign", "signpost", "pole", "lamppost", "streetlight", "hydrant", "bollard"})),
    ("furniture", frozenset({"chair", "table", "couch", "sofa", "desk", "stool", "furniture"})),
]


def _category_for(caption: str) -> str:
    words = set(caption.lower().replace(",", " ").replace(".", " ").replace("'", " ").split())
    for category, keywords in _CATEGORY_KEYWORDS:
        if keywords & words:
            return category
    return "other"


class ObjectDistributionStage(PipelineStage):
    """
    Groups placed scene objects by semantic category and computes a spatial
    distribution per category for use by ScenePopulationStage.

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i}
            (metadata must include world_position / world_width / world_height,
             written by SceneGenerationStage, and caption from classification)
    Writes: ContextKey.OBJECT_DISTRIBUTION
            {category: {indices, positions_xz, mean_width, mean_height, typical_spacing}}
    """

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects — skipping distribution")
            return context

        task = self.create_progress(object_count, "Building object distribution...")

        # Collect per-category lists
        buckets: dict[str, dict] = {}

        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}
            self.advance_progress(task)

            if metadata.get("class") == "environment":
                continue

            pos = metadata.get("world_position")
            w   = metadata.get("world_width")
            h   = metadata.get("world_height")
            if pos is None or w is None or h is None:
                continue

            caption  = metadata.get("caption", "")
            category = _category_for(caption)

            if category not in buckets:
                buckets[category] = {"indices": [], "positions_xz": [], "widths": [], "heights": []}

            buckets[category]["indices"].append(idx)
            buckets[category]["positions_xz"].append([pos[0], pos[2]])
            buckets[category]["widths"].append(w)
            buckets[category]["heights"].append(h)

        self.finish_progress(task)

        distribution: dict[str, dict] = {}
        for category, data in buckets.items():
            positions = np.array(data["positions_xz"], dtype=np.float32)
            mean_w = float(np.mean(data["widths"]))
            mean_h = float(np.mean(data["heights"]))

            if len(positions) >= 2:
                tree = KDTree(positions)
                dists, _ = tree.query(positions, k=2)
                typical_spacing = float(np.median(dists[:, 1]))
            else:
                # Single object: estimate spacing from its footprint
                typical_spacing = float(max(mean_w, mean_h)) * 2.0

            distribution[category] = {
                "indices":         data["indices"],
                "positions_xz":    positions.tolist(),
                "mean_width":      mean_w,
                "mean_height":     mean_h,
                "typical_spacing": typical_spacing,
            }
            self.log_info(
                f"  {category}: {len(data['indices'])} objects, "
                f"typical spacing {typical_spacing:.2f} m, "
                f"mean size {mean_w:.2f} x {mean_h:.2f} m"
            )

        context.add_object(ContextKey.OBJECT_DISTRIBUTION, distribution)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(ContextKey.OBJECT_DISTRIBUTION) is not None

    def model_names(self) -> list[str]:
        return []
