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


class VideoGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        width: int = 1536,
        height: int = 1024,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 30,
        motion_prompt: str = DEFAULT_MOTION_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        quantization: Optional[str] = None,
        camera_lora_strength: float = 1.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.width = width
        self.height = height
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

    Input key   (SemanticKey.INPUT)   → ContextKey.INPUT           (Image)
    Caption key (SemanticKey.CAPTION) → ContextKey.INPUT_CAPTION   (str)
    Output key  (SemanticKey.OUTPUT)  → ContextKey.GENERATED_VIDEO (Video)

    Config:
      width, height, num_frames, frame_rate, num_inference_steps — generation params.
        num_frames must be 8k+1; width/height must be divisible by 64.
      motion_prompt   — appended to the caption to keep the camera static.
      negative_prompt — steers away from camera movement and quality artifacts.
      quantization    — None (default), "fp8-cast", or "fp8-scaled-mm".
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
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT,
            SemanticKey.CAPTION: ContextKey.INPUT_CAPTION,
            SemanticKey.OUTPUT: ContextKey.GENERATED_VIDEO,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, caption_key, output_key = self._resolved_keys()

        input_image = context.input_image(input_key)
        if input_image is None:
            self.log_info("No input image, skipping")
            return context

        caption = context.object(caption_key) or ""
        prompt = f"{caption} {self.config.motion_prompt}".strip()

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
            width=self.config.width,
            height=self.config.height,
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
                "Resolution": f"{self.config.width} × {self.config.height} px",
                "Frames": str(self.config.num_frames),
                "Size": f"{video.size_bytes / (1024 * 1024):.1f} MB",
            },
        )

    def clean_up(self):
        if self._generator is not None:
            self._generator.close()
            self._generator = None
        super().clean_up()
