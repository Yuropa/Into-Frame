from path_utils import add_project_paths, lib_path, checkpoints_path, add_system_path
add_project_paths()

lux_dit_lib = lib_path() / "LuxDiT"
add_system_path(lux_dit_lib)

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from omegaconf import OmegaConf, ListConfig

from src.models.custom_cogvideox_transformer_3d import CustomCogVideoXTransformer3DModel
from src.models.custom_autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDPMScheduler
from src.pipelines.pipeline_cogvideox_rgbxenv import RGBXEnvCogVideoXPipeline
from src.data.rendering_utils import resize_crop, envmap_vec
from luxdit_config import TrainingConfig
from remote_connection.remote_server import RemoteServer


class LuxDiTServer(RemoteServer):
    def setup(self):
        checkpoint_dir = checkpoints_path() / "LuxDiT"

        # Locate config: try checkpoint dir first, then lib configs/
        config_files = sorted(checkpoint_dir.glob("*.yaml"))
        if not config_files:
            config_files = sorted((lux_dit_lib / "configs").glob("*.yaml"))
        if not config_files:
            raise FileNotFoundError(
                f"No LuxDiT config (.yaml) found in {checkpoint_dir} or {lux_dit_lib / 'configs'}"
            )
        config_path = config_files[0]
        print(f"Loading LuxDiT config from {config_path}")

        schema = OmegaConf.structured(TrainingConfig)
        cfg = OmegaConf.load(config_path)
        for key in set(cfg.keys()) - set(schema.keys()):
            OmegaConf.update(schema, key, None, force_add=True)
        self.cfg = OmegaConf.merge(schema, cfg)

        pipeline_cfg = self.cfg.model_pipeline

        # Resolve VAE/scheduler paths: fall back to checkpoint dir if not set
        if pipeline_cfg.get("vae_path", None) is None:
            OmegaConf.update(pipeline_cfg, "vae_path", str(checkpoint_dir), force_add=True)
        if pipeline_cfg.get("transformer_path", None) is None:
            OmegaConf.update(pipeline_cfg, "transformer_path", str(checkpoint_dir), force_add=True)

        weight_dtype = torch.bfloat16

        transformer_kwargs = dict(pipeline_cfg.get("transformer_kwargs", {}))
        for k, v in transformer_kwargs.items():
            if isinstance(v, (list, ListConfig)):
                transformer_kwargs[k] = tuple(v)

        vae = AutoencoderKLCogVideoX.from_pretrained(
            pipeline_cfg.vae_path,
            subfolder="vae",
            torch_dtype=weight_dtype,
            device=self.device,
        )

        # Always use the locally-downloaded fine-tuned transformer
        transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
            str(checkpoint_dir) + "/luxdit_video",
            subfolder="transformer",
            torch_dtype=weight_dtype,
            device=self.device,
            **transformer_kwargs,
        )

        base_path = str(self.cfg.pretrained_model_name_or_path) if hasattr(self.cfg, "pretrained_model_name_or_path") else str(checkpoint_dir)
        noise_scheduler = CogVideoXDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

        pipeline = RGBXEnvCogVideoXPipeline(
            vae=vae,
            tokenizer=None,
            text_encoder=None,
            transformer=transformer,
            scheduler=noise_scheduler,
        )
        self.pipeline = pipeline.to(self.device)
        self.pipeline_cfg = pipeline_cfg
        print(f"LuxDiT pipeline loaded from {checkpoint_dir}")

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "estimate":
            return self.estimate(input)
        raise ValueError(f"Unknown action: {action}")

    def estimate(self, input_image: PILImage.Image) -> dict:
        pipeline_cfg = self.pipeline_cfg

        resolution = (480, 720)      # (H, W) input to the model
        env_resolution = (128, 256)  # (H, W) output environment map

        img = np.array(input_image.convert("RGB")).astype(np.float32) / 255.0
        img = resize_crop(img, resolution)

        env_nrm_raw = envmap_vec(env_resolution)
        env_nrm = (env_nrm_raw * 0.5 + 0.5).numpy()

        vid_example = {
            "rgb": img[None, :],       # (1, H, W, 3)
            "env_nrm": env_nrm[None, :],  # (1, H, W, 3)
        }

        _target_labels = list(pipeline_cfg.target_image)
        cond_labels = dict(pipeline_cfg.cond_images)
        cond_labels.pop("env_ldr", None)
        cond_labels.pop("env_log", None)

        _additional_target_labels = list(pipeline_cfg.get("additional_target_image", []))
        additional_cond_labels = pipeline_cfg.get("additional_cond_labels", None)

        denoise_type = pipeline_cfg.get("denoise_type", "additional_hidden_states")
        separate_timesteps = pipeline_cfg.get("separate_timesteps", False)
        additional_rope_time_only = pipeline_cfg.get("additional_rope_time_only", True)
        text_prompt = pipeline_cfg.get("text_prompt", None)

        if denoise_type == "additional_hidden_states":
            target_labels = _additional_target_labels
        else:
            target_labels = _target_labels

        _, cond_images = self.pipeline.example2input(vid_example, target_labels, cond_labels)

        self.report_progress(0.1, "Running LuxDiT inference...")

        height, width = resolution
        with torch.no_grad():
            pred = self.pipeline(
                prompt=text_prompt,
                cond_images=cond_images,
                cond_mapping=cond_labels,
                guidance_scale=2.5,
                use_dynamic_cfg=False,
                num_inference_steps=50,
                height=height,
                width=width,
                num_frames=1,
                additional_cond_labels=additional_cond_labels,
                attention_kwargs=None,
                denoise_type=denoise_type,
                target_labels=_target_labels,
                additional_target_labels=_additional_target_labels,
                num_latents=len(target_labels),
                separate_timesteps=separate_timesteps,
                additional_rope_time_only=additional_rope_time_only,
            ).frames

        # pred[target_idx][batch_idx] → List[PIL.Image] (one per frame)
        frames_ldr = pred[0][0]
        frames_log = pred[1][0]

        self.report_progress(1.0, "Done")
        return {"ldr": frames_ldr[0], "log": frames_log[0]}


if __name__ == "__main__":
    LuxDiTServer.run()
