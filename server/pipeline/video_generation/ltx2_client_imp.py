from path_utils import add_project_paths, checkpoints_path
add_project_paths()

import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_LORA_STRENGTH, LTX_2_3_PARAMS
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from remote_connection.remote_server import RemoteServer

# Filenames as downloaded by scripts/setup.sh's download_ltx2_models() into checkpoints/ltx2.
CHECKPOINT_FILE = "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA_FILE = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
SPATIAL_UPSAMPLER_FILE = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
CAMERA_CONTROL_STATIC_LORA_FILE = "ltx-2-19b-lora-camera-control-static.safetensors"
GEMMA_SUBDIR = "gemma-3-12b"


class LTX2Server(RemoteServer):
    def setup(self):
        ckpt_dir = checkpoints_path() / "ltx2"
        checkpoint_path = str(ckpt_dir / CHECKPOINT_FILE)

        quantization_name = os.environ.get("LTX2_QUANTIZATION")
        quantization = (
            QuantizationKind(quantization_name).to_policy(checkpoint_path=checkpoint_path)
            if quantization_name
            else None
        )

        # Applied to both stages (unlike distilled_lora, which is stage-2-only) to lock the
        # camera off, reinforcing VideoGenerationStage's "static camera" prompt instruction.
        # Strength 0 drops it entirely — set LTX2_CAMERA_LORA_STRENGTH=0 to disable.
        camera_lora_strength = float(os.environ.get("LTX2_CAMERA_LORA_STRENGTH", DEFAULT_LORA_STRENGTH))
        loras = (
            [
                LoraPathStrengthAndSDOps(
                    str(ckpt_dir / CAMERA_CONTROL_STATIC_LORA_FILE),
                    camera_lora_strength,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]
            if camera_lora_strength > 0
            else []
        )

        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=[
                LoraPathStrengthAndSDOps(
                    str(ckpt_dir / DISTILLED_LORA_FILE),
                    DEFAULT_LORA_STRENGTH,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ],
            spatial_upsampler_path=str(ckpt_dir / SPATIAL_UPSAMPLER_FILE),
            gemma_root=str(ckpt_dir / GEMMA_SUBDIR),
            loras=loras,
            device=self.device,
            quantization=quantization,
        )
        print(f"LTX-2 pipeline loaded from {ckpt_dir} (camera_lora={camera_lora_strength})")

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "generate":
            return self._generate(temp_path, input)
        raise ValueError(f"Unknown action: {action}")

    def _generate(self, temp_path: Path, input: dict) -> str:
        image: PILImage.Image = input["image"]
        prompt = str(input["prompt"])
        negative_prompt = str(input.get("negative_prompt", ""))
        seed = int(input.get("seed", 0))
        width = int(input.get("width", LTX_2_3_PARAMS.stage_2_width))
        height = int(input.get("height", LTX_2_3_PARAMS.stage_2_height))
        num_frames = int(input.get("num_frames", LTX_2_3_PARAMS.num_frames))
        frame_rate = float(input.get("frame_rate", LTX_2_3_PARAMS.frame_rate))
        num_inference_steps = int(input.get("num_inference_steps", LTX_2_3_PARAMS.num_inference_steps))

        image_path = temp_path / "condition.png"
        image.convert("RGB").save(str(image_path))

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        self.report_progress(0.05, "Running LTX-2 two-stage generation…")
        with torch.inference_mode():
            video, audio = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                video_guider_params=LTX_2_3_PARAMS.video_guider_params,
                audio_guider_params=LTX_2_3_PARAMS.audio_guider_params,
                images=[ImageConditioningInput(path=str(image_path), frame_idx=0, strength=1.0)],
                tiling_config=tiling_config,
            )

        self.report_progress(0.9, "Encoding video…")
        output_path = temp_path / "output.mp4"
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )

        self.report_progress(1.0, "Done")
        return str(output_path)


if __name__ == "__main__":
    LTX2Server.run()
