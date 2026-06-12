import json
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_detection.grounding_dino import GroundingDino
from util.image_utils import Image


class ObjectDetectionStage(PipelineStage):
    """
    Localises every RAM++ tag in the input image using Grounding DINO, appending
    new crop_{i} / metadata_{i} entries after the existing SAM2 detections.

    Reads:  ContextKey.RECOGNIZE_TAGS (pipe-separated str), SemanticKey.INPUT image,
            ContextKey.OBJECT_COUNT (existing SAM2 count)
    Writes: crop_{i}, metadata_{i} (type, score, box, source='grounding_dino')
            for each new detection; updated ContextKey.OBJECT_COUNT
    Debug:  self.output/detections.json — list of new detections with label, box, score
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._detector = None

    def _input_key(self):
        return self._resolve_key(SemanticKey.INPUT, ContextKey.INPUT)

    def run(self, context: PipelineContext) -> PipelineContext:
        tags_str = context.input_object(ContextKey.RECOGNIZE_TAGS)
        if not tags_str:
            self.log_info("No recognition tags — skipping detection")
            return context

        input_image = context.input_image(self._input_key())
        if input_image is None:
            self.log_info("No input image — skipping detection")
            return context

        tags = [t.strip() for t in tags_str.split("|") if t.strip()]
        existing_count = context.input_object(ContextKey.OBJECT_COUNT) or 0

        task = self.create_progress(2, "Detecting objects…")
        if self._detector is None:
            self._detector = GroundingDino(self.device)
        self.advance_progress(task)

        detections = self._detector.detect(
            input_image.rgb(),
            tags,
            self.temp or self.output,
            on_progress=self.make_progress_callback(task),
        )
        self.advance_progress(task)
        self.finish_progress(task)

        img_w, img_h = input_image.size
        next_idx = existing_count
        debug_entries = []

        for det in detections:
            x, y, w, h = det["box"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = Image(input_image.rgb().crop((x1, y1, x2, y2)))
            metadata = {
                "box": [float(x), float(y), float(w), float(h)],
                "score": float(det["score"]),
                "type": det["label"],
                "source": "grounding_dino",
            }

            context.add_image(f"crop_{next_idx}", crop)
            context.add_object(f"metadata_{next_idx}", metadata)

            entry = {"index": next_idx, "label": det["label"], "score": round(det["score"], 4), "box": [round(v, 1) for v in [x, y, w, h]]}
            debug_entries.append(entry)
            self.log_info(f"  crop_{next_idx}: '{det['label']}' score={det['score']:.3f}")
            next_idx += 1

        added = next_idx - existing_count
        context.add_object(ContextKey.OBJECT_COUNT, next_idx)
        context.add_object("detection_complete", True)

        if self.output is not None and debug_entries:
            debug_path = self.output / "detections.json"
            with open(debug_path, "w") as f:
                json.dump(debug_entries, f, indent=2)

        self.log_info(f"Added {added} detections (SAM2: {existing_count}, total: {next_idx})")
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object("detection_complete") is not None

    def model_names(self) -> list[str]:
        return GroundingDino.model_names()

    def clean_up(self):
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        super().clean_up()
