import json
from logging import Logger
from typing import Any

import torch
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_detection.grounding_dino import GroundingDino
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.segmentation_result import SegmentationResult
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image


class ObjectDetectionConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        # Same size-filter reasoning as SegmentationConfiguration (see
        # pipeline/segmentation/segmentation.py) -- Grounding DINO/box-prompted SAM2
        # crops outside this bounding-box-area range are dropped before being stored.
        min_area_fraction: float = 0.0003,
        max_area_fraction: float = 0.20,
        confident_score: float = 0.45,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        # Grounding DINO score at or above which a detection is trusted for its
        # identity, not just its position -- i.e. allowed to anchor a visual
        # bucket in ObjectCategoryClusteringStage. Below it the detection is
        # marked low_confidence and has to earn its class by DINOv2 similarity to
        # a confident crop, the same bar a weak CLIP typing result faces.
        #
        # 0.45 sits above the detector's own box threshold (which decides "is
        # there an object here at all") but below the scores a clean, unambiguous
        # detection reaches -- on the Rainier capture the placed flowers scored
        # 0.35-0.55, straddling it, which is the intended behaviour rather than a
        # coincidence: those weaker boxes are real objects whose *labels* are not
        # independently trustworthy.
        self.confident_score = confident_score


class ObjectDetectionStage(PipelineStage):
    """
    Localises every RAM++ tag in the input image using Grounding DINO, masks each
    detection with box-prompted SAM2 (Grounding DINO only gives boxes), and appends
    new crop_{i} / metadata_{i} entries after the existing SAM2 detections.

    Input key (SemanticKey.INPUT) → ContextKey.INPUT (Image, default) or a
    Panorama at the same key — e.g. point it at ContextKey.PANORAMA to detect
    objects across the full 360° panorama instead of the narrow input photo
    (GroundingDinoServer tiles internally in that case — see grounding_dino_imp.py).

    Reads:  ContextKey.RECOGNIZE_TAGS (pipe-separated str), SemanticKey.INPUT image,
            ContextKey.OBJECT_COUNT (existing SAM2 count)
    Writes: crop_{i} (masked RGBA crop), metadata_{i} (class, score, box, source='grounding_dino')
            for each new detection; updated ContextKey.OBJECT_COUNT
    Debug:  self.output/detections.json — list of new detections with label, box, score

    Config: min_area_fraction/max_area_fraction (see ObjectDetectionConfiguration).
    """

    @classmethod
    def config_class(cls) -> type[ObjectDetectionConfiguration]:
        return ObjectDetectionConfiguration

    def __init__(self, config: ObjectDetectionConfiguration) -> None:
        super().__init__(config)
        self._detector = None
        self._seg = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def _input_key(self):
        return self._resolve_key(SemanticKey.INPUT, ContextKey.INPUT)

    def _resolve_source(self, context: PipelineContext, input_key) -> "Image | None":
        """Image and Panorama both expose .width/.height/.rgb()/.image, so either
        can be the detection source — try Image (the common case) before Panorama,
        wrapping a Panorama into a plain Image since Panorama has no .rgba(), which
        masked_images() below needs."""
        image = context.input_image(input_key)
        if image is not None:
            return image
        panorama = context.input_panorama(input_key)
        if panorama is not None:
            return Image(panorama.image)
        return None

    def run(self, context: PipelineContext) -> PipelineContext:
        tags_str = context.input_object(ContextKey.RECOGNIZE_TAGS)
        if not tags_str:
            self.log_info("No recognition tags — skipping detection")
            return context

        input_image = self._resolve_source(context, self._input_key())
        if input_image is None:
            self.log_info("No input image or panorama — skipping detection")
            return context

        tags = [t.strip() for t in tags_str.split("|") if t.strip()]
        existing_count = context.input_object(ContextKey.OBJECT_COUNT) or 0

        task = self.create_progress(3, "Detecting objects…")
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

        img_w, img_h = input_image.size

        # Grounding DINO only gives boxes, not masks — clamp each to the image and
        # SAM2-mask it (box-prompted) so these crops are cut out the same way the
        # SAM2 automatic detections are, instead of coming through as plain
        # rectangles that would render as opaque bounding boxes on a billboard.
        valid_dets = []
        clamped_boxes = []
        for det in detections:
            x, y, w, h = det["box"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            valid_dets.append(det)
            clamped_boxes.append([x1, y1, x2 - x1, y2 - y1])

        masks = []
        if clamped_boxes:
            if self._seg is None:
                self._seg = ImageSeg(self.preferred_device)
            masks = self._seg.segment_boxes(
                input_image, clamped_boxes, self.temp,
                on_progress=self.make_progress_callback(task),
            )
        self.advance_progress(task)
        self.finish_progress(task)

        next_idx = existing_count
        debug_entries = []
        filtered = 0

        cfg: ObjectDetectionConfiguration = self.config
        img_area = img_w * img_h
        min_area = cfg.min_area_fraction * img_area
        max_area = cfg.max_area_fraction * img_area

        if masks:
            result = SegmentationResult(masks=masks, boxes=clamped_boxes, scores=[d["score"] for d in valid_dets])
            for det, crop in zip(valid_dets, result.masked_images(input_image)):
                # Use crop.box (== the clamped box the mask/crop was actually generated
                # from), not det["box"] (Grounding DINO's raw, possibly out-of-image
                # box) -- downstream depth sampling and world size/position estimation
                # key off this box, and it must describe the same pixels as crop.image.
                x, y, w, h = crop.box
                if w * h < min_area or w * h > max_area:
                    filtered += 1
                    continue

                metadata = {
                    "box": [float(x), float(y), float(w), float(h)],
                    "score": float(det["score"]),
                    "class": det["label"],
                    "source": "grounding_dino",
                    # These detections are minted AFTER ObjectTypingStage has run,
                    # so they never receive its 'low_confidence' verdict. That
                    # absence is not neutral: ObjectCategoryClusteringStage reads
                    # a missing flag as falsy, i.e. fully confident, so a weak
                    # box would silently get to anchor a visual bucket with more
                    # authority than a CLIP-verified crop. (Before the typing
                    # confidence metric was fixed, that accident was the only
                    # reason anything survived clustering at all -- which is
                    # exactly why it can't be left implicit now that it doesn't
                    # have to carry the scene.)
                    #
                    # Grounding DINO's own score is real image-grounded evidence,
                    # just from a different model, so it answers the question
                    # directly rather than deferring to a pass that will never
                    # see these crops.
                    "low_confidence": float(det["score"]) < cfg.confident_score,
                }

                context.add_image(f"crop_{next_idx}", crop.image)
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

        if filtered:
            self.log_info(f"  Filtered {filtered} detection(s) outside size bounds")
        self.log_info(f"Added {added} detections (SAM2: {existing_count}, total: {next_idx})")
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object("detection_complete") is not None

    def model_names(self) -> list[str]:
        return GroundingDino.model_names() + ImageSeg.model_names()

    def clean_up(self):
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        super().clean_up()
