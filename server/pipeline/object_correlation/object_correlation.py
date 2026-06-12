import colorsys
import json
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_correlation.object_correlation_result import ObjectCorrelationResult, ObjectGroupStats


def _iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two [x, y, w, h] boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _category_colors(categories: list[str]) -> dict[str, tuple[int, int, int]]:
    """Assign a visually distinct RGB color to each category."""
    n = len(categories)
    colors = {}
    for i, cat in enumerate(sorted(categories)):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
        colors[cat] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


class ObjectCorrelationStage(PipelineStage):
    """
    Groups all detected objects (SAM2 + Grounding DINO) by type label.

    Grounding DINO detections that substantially overlap an existing SAM2 detection
    are dropped to avoid double-counting (IoU threshold: 0.5).

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i}, ContextKey.INPUT (for debug image)
    Writes: ContextKey.OBJECT_CORRELATION (ObjectCorrelationResult)
    Debug:  self.output/stats.json        — per-category counts and indices
            self.output/debug.png         — input image with per-category colored boxes
    """

    _IOU_DEDUP_THRESHOLD = 0.5

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to correlate, skipping")
            return context

        task = self.create_progress(object_count, "Correlating objects...")

        # Separate SAM2 vs Grounding DINO detections
        sam2_meta = {}
        gdino_meta = {}
        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}
            if metadata.get("source") == "grounding_dino":
                gdino_meta[idx] = metadata
            else:
                sam2_meta[idx] = metadata

        # Dedup: drop GDINO detections that overlap a SAM2 detection
        sam2_boxes = [m.get("box") for m in sam2_meta.values() if m.get("box")]
        deduplicated = 0
        surviving_gdino = {}
        for idx, metadata in gdino_meta.items():
            box = metadata.get("box")
            if box and any(_iou(box, sb) >= self._IOU_DEDUP_THRESHOLD for sb in sam2_boxes):
                self.log_info(f"  crop_{idx}: '{metadata.get('type', '?')}' duplicate of SAM2 detection — dropped")
                deduplicated += 1
            else:
                surviving_gdino[idx] = metadata

        # Build correlation groups from all surviving objects
        result = ObjectCorrelationResult(deduplicated_count=deduplicated)
        all_meta = {**sam2_meta, **surviving_gdino}

        for idx in sorted(all_meta):
            metadata = all_meta[idx]
            obj_type = metadata.get("type") or "unknown"
            if obj_type not in result.groups:
                result.groups[obj_type] = ObjectGroupStats(object_type=obj_type)
            result.groups[obj_type].indices.append(idx)
            self.advance_progress(task)

        # Advance remaining progress slots for deduplicated items
        for _ in range(deduplicated):
            self.advance_progress(task)

        context.add_object_correlation(ContextKey.OBJECT_CORRELATION, result)
        self.finish_progress(task)

        # Log summary
        for obj_type, stats in sorted(result.groups.items()):
            self.log_info(f"  {obj_type}: {stats.count} object(s) — indices {stats.indices}")
        if deduplicated:
            self.log_info(f"  {deduplicated} GDINO detection(s) dropped as SAM2 duplicates")

        self._write_debug(context, result)
        return context

    def _write_debug(self, context: PipelineContext, result: ObjectCorrelationResult):
        if self.output is None:
            return

        # Stats JSON
        stats_path = self.output / "stats.json"
        stats = {
            "deduplicated_gdino": result.deduplicated_count,
            "categories": {
                obj_type: {"count": stats.count, "indices": stats.indices}
                for obj_type, stats in result.groups.items()
            },
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Debug overlay image
        input_image = context.input_image(ContextKey.INPUT)
        if input_image is None:
            return

        base = input_image.rgb().convert("RGBA")
        overlay = PIL.Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(overlay)
        font = PIL.ImageFont.load_default()

        colors = _category_colors(result.types())

        for obj_type, stats in result.groups.items():
            r, g, b = colors[obj_type]
            fill = (r, g, b, 50)
            outline = (r, g, b, 220)

            for idx in stats.indices:
                metadata = context.input_object(f"metadata_{idx}") or {}
                box = metadata.get("box")
                if not box:
                    continue
                x, y, w, h = box
                draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=2)
                draw.text((x + 4, y + 4), obj_type, fill=(r, g, b, 255), font=font)

        composite = PIL.Image.alpha_composite(base, overlay).convert("RGB")
        debug_path = self.output / "debug.png"
        composite.save(debug_path)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object_correlation(ContextKey.OBJECT_CORRELATION) is not None
