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

        # LuxDiT's own HDR merger. The two maps the diffusion pipeline emits are
        # both 8-bit: env_ldr is tonemapped and env_log is a *learned* log-domain
        # encoding, and there is no closed-form inverse for the latter -- the
        # repo's hdr_merger.py reconstructs radiance with this small trained
        # network, not a formula. Skipping it (as this server previously did)
        # leaves everything downstream reasoning about lighting from compressed
        # 8-bit pixels, where the sun is only ~6x the map mean instead of the
        # ~30x+ it really is, so no honest intensity can be recovered.
        #
        # Weights ship in the same nvidia/LuxDiT snapshot setup.sh already
        # downloads (the `hdr_merge_mlp` folder), so this costs no extra setup.
        #
        # hdr_status records WHY the merge did or didn't happen and travels back to
        # the caller alongside the maps. This server runs in its own conda env as a
        # subprocess, so the print()s here go to a stdout the pipeline log never
        # captures, which made an LDR-only fallback silent from the pipeline's side:
        # all five sample captures ran without HDR -- shipping a key light 2.4x-12x
        # under Unity's nominal 1.0 -- and the only trace was "no HDR merge", with
        # no way to tell missing weights from unloadable ones from a rejected result.
        self.hdr_status = "not attempted"
        self.hdr_model = None
        merger_dir = checkpoint_dir / "hdr_merge_mlp"
        if merger_dir.is_dir():
            try:
                from src.models.hdr_model import HDR_MLP
                self.hdr_model = HDR_MLP.from_pretrained(str(merger_dir)).to(self.device).eval()
                self.hdr_status = f"merger loaded from {merger_dir}"
            except Exception as e:
                self.hdr_status = f"merger at {merger_dir} failed to load: {e}"
        else:
            self.hdr_status = (
                f"no merger at {merger_dir} — the nvidia/LuxDiT snapshot should "
                f"contain an hdr_merge_mlp/ folder; re-run download_weights.py"
            )
        print(f"LuxDiT HDR: {self.hdr_status}")

    def _merger_samples_per_branch(self) -> int:
        """How many values of one map the merger takes per sample.

        HDR_MLP concatenates its two arguments (`torch.cat((x_ldr, x_hdr), dim=-1)`)
        before its first Linear, so that layer's in_features counts BOTH branches and
        half of it is the width each map must be reshaped to. Read from the loaded
        weights rather than assumed: this file previously hard-coded a per-channel
        scalar network (in_features 2 -> 1 value per branch) on the strength of a
        docstring, and the released checkpoint's first Linear is (6, 64) -- it wants a
        whole RGB triple from each map. Every lighting estimate since has died on
        `mat1 and mat2 shapes cannot be multiplied (98304x2 and 6x64)` and silently
        fallen back to an LDR-only sun ~half of Unity's nominal daylight.

        Returns 0 when no Linear can be found, which the caller reports rather than
        guessing at.
        """
        for module in self.hdr_model.modules():
            if isinstance(module, torch.nn.Linear):
                return max(module.in_features // 2, 0)
        return 0

    def _merge_hdr(self, ldr_img: PILImage.Image, log_img: PILImage.Image):
        """Reconstruct linear HDR radiance from the (env_ldr, env_log) pair.

        Mirrors hdr_merger.py: both maps are normalised to [-1, 1] and fed to the
        merger together, flattened to (N, C) where C is whatever the loaded weights
        ask for -- 3 for the released per-pixel-RGB checkpoint, 1 for a per-channel
        scalar variant. See _merger_samples_per_branch.

        Returns float32 (H, W, 3) linear radiance, or None if anything about the
        result doesn't look like radiance -- a silently wrong reconstruction here
        would poison every lighting decision downstream, so this fails loudly and
        lets the caller fall back rather than shipping plausible-looking garbage.
        """
        if self.hdr_model is None:
            return None
        try:
            ldr = np.asarray(ldr_img.convert("RGB"), dtype=np.float32) / 255.0 * 2.0 - 1.0
            log = np.asarray(log_img.convert("RGB"), dtype=np.float32) / 255.0 * 2.0 - 1.0
            if ldr.shape != log.shape:
                self.hdr_status = f"merge skipped: shape mismatch {ldr.shape} vs {log.shape}"
                return None

            channels = self._merger_samples_per_branch()
            if channels <= 0 or ldr.size % channels != 0:
                self.hdr_status = (
                    f"merge skipped: merger takes {channels} value(s) per branch, "
                    f"which does not divide a {ldr.shape} map"
                )
                return None

            shape = ldr.shape
            with torch.no_grad():
                x_ldr = torch.from_numpy(ldr.reshape(-1, channels)).to(self.device)
                x_log = torch.from_numpy(log.reshape(-1, channels)).to(self.device)
                out = self.hdr_model(x_ldr, x_log).float().cpu().numpy()

            # A merger whose output width doesn't match its input width would reshape
            # into a differently-sized map and silently scramble the radiance across
            # pixels rather than fail, so check before trusting it.
            if out.size != ldr.size:
                self.hdr_status = (
                    f"merge rejected: merger returned {out.size} value(s) for a "
                    f"{shape} map ({ldr.size} expected)"
                )
                return None
            hdr = out.reshape(shape)

            if not np.isfinite(hdr).all() or hdr.min() < 0.0 or hdr.max() <= 0.0:
                self.hdr_status = f"merge rejected: range [{hdr.min()}, {hdr.max()}] is not radiance"
                return None
            self.hdr_status = (
                f"merged: radiance range [{hdr.min():.4f}, {hdr.max():.2f}], "
                f"peak/mean {hdr.max() / max(float(hdr.mean()), 1e-9):.1f}x"
            )
            return hdr.astype(np.float32)
        except Exception as e:
            self.hdr_status = f"merge failed: {e}"
            return None

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

        self.report_progress(0.1, "Running LuxDiT inference…")

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

        self.report_progress(0.9, "Merging HDR…")
        hdr = self._merge_hdr(frames_ldr[0], frames_log[0])

        self.report_progress(1.0, "Done")
        # hdr is None when the merger is unavailable or produced something that
        # isn't radiance; SceneLighting falls back to the compressed maps then.
        return {"ldr": frames_ldr[0], "log": frames_log[0], "hdr": hdr,
                "hdr_status": self.hdr_status}


if __name__ == "__main__":
    LuxDiTServer.run()
