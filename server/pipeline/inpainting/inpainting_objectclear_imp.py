from path_utils import add_project_paths, add_system_path, lib_path
add_project_paths()
add_system_path(lib_path() / "ObjectClear")

import torch
from pathlib import Path
from typing import Any
from PIL import Image as PILImage
from remote_connection.remote_server import RemoteServer


def _resize_short_side(image: PILImage.Image, target: int, resample: int) -> PILImage.Image:
    w, h = image.size
    short = min(w, h)
    if short == target:
        return image
    scale = target / short
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return image.resize((new_w, new_h), resample)


class ObjectClearInpainter(RemoteServer):
    def setup(self):
        from objectclear.pipelines import ObjectClearPipeline

        use_fp16 = self.device.type == "cuda"
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        variant = "fp16" if use_fp16 else None

        self.pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=torch_dtype,
            apply_attention_guided_fusion=True,
            variant=variant,
        )
        self.pipe.to(self.device)

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "inpaint":
            return self.inpaint(
                input["image"],
                input["mask"],
                prompt=input.get("prompt", "remove the instance of object"),
                num_inference_steps=input.get("num_inference_steps", 30),
                guidance_scale=input.get("guidance_scale", 2.5),
                seed=input.get("seed", 0),
            )
        raise ValueError(f"Unknown action: {action}")

    def inpaint(
        self,
        image: PILImage.Image,
        mask: PILImage.Image,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> PILImage.Image:
        orig_w, orig_h = image.size

        image_r = _resize_short_side(image.convert("RGB"), 512, PILImage.BICUBIC)
        mask_r = _resize_short_side(mask.convert("L"), 512, PILImage.NEAREST)
        w, h = image_r.size

        generator = torch.Generator(device=self.device).manual_seed(seed)

        self.report_progress(0.1, "Running ObjectClear inference…")
        result = self.pipe(
            prompt=prompt,
            image=image_r,
            mask_image=mask_r,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=h,
            width=w,
            return_attn_map=False,
        )
        self.report_progress(1.0, "Done")

        out = result.images[0]
        if out.size != (orig_w, orig_h):
            out = out.resize((orig_w, orig_h), PILImage.LANCZOS)
        return out.convert("RGB")


if __name__ == "__main__":
    ObjectClearInpainter.run()
