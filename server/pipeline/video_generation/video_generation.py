import math
import torch
from logging import Logger
from typing import Any, Optional

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.video_generation.ltx2_client import LTX2VideoGenerator
from util.device_utils import DeviceStrategy, preferred_device

# Appended to the CaptioningStage caption so the video reads as a living photo rather
# than a moving-camera shot: locked-off framing with only ambient environmental motion.
DEFAULT_MOTION_PROMPT = (
    "The camera is completely static on a locked tripod, with no panning, zooming, "
    "dolly, or handheld movement. Only subtle natural motion animates the scene: a "
    "gentle breeze stirs leaves and grasses, light ripples cross any visible water, "
    "and clouds drift almost imperceptibly across the sky."
)

DEFAULT_NEGATIVE_PROMPT = (
    "camera movement, camera panning, zooming, dolly movement, tracking shot, "
    "handheld shake, jittery motion, blurry, low quality, distorted, warped "
    "geometry, artifacts, watermark, text overlay"
)

# LTX-2 two-stage pipelines require height/width divisible by 64 (assert_resolution).
RESOLUTION_DIVISOR = 64

# The unquantized ~22B checkpoint runs a 32GB card out of memory on its own; fp8-cast
# keeps generation working on that class of card without a noticeable quality hit.
DEFAULT_QUANTIZATION = "fp8-cast"


def _round_to_divisor(value: float) -> int:
    return max(RESOLUTION_DIVISOR, round(value / RESOLUTION_DIVISOR) * RESOLUTION_DIVISOR)


def resolve_video_resolution(
    source_width: int,
    source_height: int,
    target_pixels: int,
    min_side: int,
    max_side: int,
) -> tuple[int, int]:
    """Pick an output width/height close to the source image's own aspect ratio.

    LTX-2's image conditioning resizes-to-fill then center-crops to the exact
    output resolution, so a fixed 16:9-ish default would silently crop the sides
    off a portrait photo or the top/bottom off a panorama. Instead, size the
    output to roughly `target_pixels` total (the model's native/tuned area, ~1.5MP
    for LTX-2.3) while matching the source aspect ratio, then round each side to
    the nearest multiple of 64 and clamp into [min_side, max_side] to bound
    compute/VRAM on extreme aspect ratios (e.g. a 360° equirectangular panorama).
    """
    aspect = source_width / source_height
    height = math.sqrt(target_pixels / aspect)
    width = height * aspect

    width = min(max(_round_to_divisor(width), min_side), max_side)
    height = min(max(_round_to_divisor(height), min_side), max_side)
    return width, height


class VideoGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        target_pixels: int = 1536 * 1024,
        min_side: int = 512,
        max_side: int = 2048,
        num_frames: int = 169,  # ~7s @ 24fps — confirmed to fit alongside fp8-cast on a 32GB card
        frame_rate: float = 24.0,
        num_inference_steps: int = 30,
        motion_prompt: str = DEFAULT_MOTION_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        quantization: Optional[str] = DEFAULT_QUANTIZATION,
        camera_lora_strength: float = 1.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.target_pixels = target_pixels
        self.min_side = min_side
        self.max_side = max_side
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.num_inference_steps = num_inference_steps
        self.motion_prompt = motion_prompt
        self.negative_prompt = negative_prompt
        self.quantization = quantization
        self.camera_lora_strength = camera_lora_strength


class VideoGenerationStage(PipelineStage):
    """
    Animates the input photograph into a short audio-video clip using LTX-2
    (TI2VidTwoStagesPipeline): a locked-off camera with only subtle natural motion
    (wind, water, clouds), so the source photo reads as a living scene rather than
    a moving-camera shot.

    The prompt is the caption produced by CaptioningStage, augmented with a fixed
    "static camera, subtle natural motion" instruction (config.motion_prompt) so no
    separate manual prompt is needed per run. Camera rigidity is additionally
    reinforced by the Camera-Control-Static LoRA (config.camera_lora_strength).

    Input key   (SemanticKey.INPUT)   → ContextKey.INPUT           (Image, default)
                                         or a Panorama at the same key — e.g. point
                                         it at ContextKey.PANORAMA to animate the
                                         generated 360° panorama instead of the raw
                                         input photo (see config_pano_video_generation.yaml).
    Caption key (SemanticKey.CAPTION) → ContextKey.INPUT_CAPTION   (str)
    Output key  (SemanticKey.OUTPUT)  → ContextKey.GENERATED_VIDEO (Video)

    Config:
      target_pixels, min_side, max_side — the output resolution isn't fixed; it's derived
        per-run from the input image's own aspect ratio (see resolve_video_resolution) so
        portrait/panoramic photos aren't force-cropped to a fixed shape. target_pixels is
        the target total area (~1.5MP default, LTX-2.3's native scale); min_side/max_side
        bound each side after rounding to a multiple of 64.
      num_frames, frame_rate, num_inference_steps — generation params.
        num_frames must be 8k+1.
      motion_prompt   — appended to the caption to keep the camera static.
      negative_prompt — steers away from camera movement and quality artifacts.
      quantization    — "fp8-cast" (default, needed to fit the ~22B checkpoint on a
                         32GB card), "fp8-scaled-mm", or None for full precision.
      camera_lora_strength — strength of the Camera-Control-Static LoRA (default 1.0,
                         full rigidity), applied on top of motion_prompt to lock the
                         camera off. 0 disables the LoRA entirely.
    """

    @classmethod
    def config_class(cls) -> type[VideoGenerationConfiguration]:
        return VideoGenerationConfiguration

    def __init__(self, config: VideoGenerationConfiguration) -> None:
        super().__init__(config)
        self._generator = None
        self._last_resolution: Optional[tuple[int, int]] = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT,
            SemanticKey.CAPTION: ContextKey.INPUT_CAPTION,
            SemanticKey.OUTPUT: ContextKey.GENERATED_VIDEO,
        })

    def _resolve_source(self, context: PipelineContext, input_key):
        """Image and Panorama both expose .width/.height/.rgb(), so either can be
        the conditioning frame — try Image (the common case) before Panorama."""
        image = context.input_image(input_key)
        if image is not None:
            return image
        return context.input_panorama(input_key)

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, caption_key, output_key = self._resolved_keys()

        input_image = self._resolve_source(context, input_key)
        if input_image is None:
            self.log_info("No input image or panorama, skipping")
            return context

        caption = context.object(caption_key) or ""
        prompt = f"{caption} {self.config.motion_prompt}".strip()

        width, height = resolve_video_resolution(
            source_width=input_image.width,
            source_height=input_image.height,
            target_pixels=self.config.target_pixels,
            min_side=self.config.min_side,
            max_side=self.config.max_side,
        )
        self._last_resolution = (width, height)
        self.log_info(f"  {input_key}: {input_image.width}x{input_image.height} → {width}x{height}")

        gen_task = self.create_progress(1, "Generating video…")
        super().clean_up()
        if self._generator is None:
            self._generator = LTX2VideoGenerator(
                self.preferred_device,
                quantization=self.config.quantization,
                camera_lora_strength=self.config.camera_lora_strength,
            )

        video_path = self._generator.generate(
            image=input_image.rgb(),
            prompt=prompt,
            negative_prompt=self.config.negative_prompt,
            temp_path=self.temp,
            seed=self.seed,
            width=width,
            height=height,
            num_frames=self.config.num_frames,
            frame_rate=self.config.frame_rate,
            num_inference_steps=self.config.num_inference_steps,
            on_progress=self.make_progress_callback(gen_task),
        )
        context.add_video(output_key, video_path)
        self.log_info(f"  {output_key}: {video_path}")

        self.finish_progress(gen_task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, output_key = self._resolved_keys()
        return context.video(output_key) is not None

    def model_names(self) -> list[str]:
        return LTX2VideoGenerator.model_names()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        _, caption_key, output_key = self._resolved_keys()
        video = context.video(output_key)
        if video is None:
            return None
        caption = context.object(caption_key) or ""
        resolution = f"{self._last_resolution[0]} × {self._last_resolution[1]} px" if self._last_resolution else "n/a"
        duration_s = self.config.num_frames / self.config.frame_rate
        return ReportSection(
            stage_name=self.name,
            title="Video Generation",
            body=(
                "The input photograph was animated into a short audio-video clip "
                "using LTX-2, conditioned on the scene caption plus a fixed "
                "instruction to keep the camera static with only subtle natural "
                "motion (wind, water, clouds)."
            ),
            stats={
                "Caption": caption,
                "Resolution": resolution,
                "Frames": f"{self.config.num_frames} ({duration_s:.1f}s @ {self.config.frame_rate:.0f}fps)",
                "Size": f"{video.size_bytes / (1024 * 1024):.1f} MB",
            },
        )

    def clean_up(self):
        if self._generator is not None:
            self._generator.close()
            self._generator = None
        super().clean_up()
