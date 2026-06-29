from enum import Enum
from util.image_utils import Image, make_context_composite
from PIL import Image as PILImage
import numpy as np
import torch


class CaptioningModel(Enum):
    BLIP = "Salesforce/blip-image-captioning-large"
    FLORENCE2 = "microsoft/Florence-2-large"


class ImageCaptioning:
    # <MORE_DETAILED_CAPTION> produces a full sentence with colour, shape, context.
    # <DETAILED_CAPTION> is slightly shorter; <CAPTION> is one-liner.
    _FLORENCE_TASK = "<MORE_DETAILED_CAPTION>"

    def __init__(self, device, model: CaptioningModel = CaptioningModel.BLIP):
        self.device = device
        self.model_type = model

        if model == CaptioningModel.BLIP:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained(model.value)
            self.net = BlipForConditionalGeneration.from_pretrained(model.value).to(device)
        else:
            self.processor, self.net = self._load_florence2(model.value)

        self.net.eval()

    def _load_florence2(self, model_id: str):
        """Load Florence-2 processor and model.

        Cached revision 21a599d4 of the Florence-2 custom modules has multiple
        bugs incompatible with newer transformers. Patch them proactively before
        every load; the patches are idempotent so this is a no-op once applied.
        """
        from transformers import AutoProcessor, AutoModelForCausalLM

        self._patch_florence2_config_cache()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)
        return processor, model

    @staticmethod
    def _patch_florence2_config_cache() -> None:
        """Fix the forced_bos_token_id bug in the cached configuration_florence2.py.

        Two known bugs in cached revision 21a599d4 of the Florence-2 custom code:
          1. configuration_florence2.py: accesses self.forced_bos_token_id before
             super().__init__() sets it — fix: use kwargs.get() instead.
          2. processing_florence2.py: accesses tokenizer.additional_special_tokens
             which doesn't exist on RobertaTokenizer in newer transformers —
             fix: guard with getattr(..., []).
        After patching files, evict stale sys.modules entries so Python re-loads
        from disk rather than the in-memory cached imports.
        """
        import glob, os, sys

        modules_root = os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules"
        )

        _CONFIG_PATCHES = [
            (
                'if self.forced_bos_token_id is None'
                ' and kwargs.get("force_bos_token_to_be_generated", False):',
                'if kwargs.get("forced_bos_token_id", None) is None'
                ' and kwargs.get("force_bos_token_to_be_generated", False):',
            ),
        ]
        _PROCESSING_PATCHES = [
            (
                'tokenizer.additional_special_tokens',
                'getattr(tokenizer, "additional_special_tokens", [])',
            ),
        ]

        def _apply(path_glob, patches):
            for fpath in glob.glob(os.path.join(modules_root, "microsoft", "Florence*", "*", path_glob)):
                with open(fpath) as f:
                    content = f.read()
                changed = False
                for old, new in patches:
                    if old in content:
                        content = content.replace(old, new)
                        changed = True
                if changed:
                    with open(fpath, "w") as f:
                        f.write(content)

        _apply("configuration_florence2.py", _CONFIG_PATCHES)
        _apply("processing_florence2.py", _PROCESSING_PATCHES)

        for key in list(sys.modules):
            if "transformers_modules" in key:
                del sys.modules[key]

    @classmethod
    def model_names(cls) -> list[str]:
        return [m.value for m in CaptioningModel]

    def _to_rgb(self, image: Image) -> PILImage.Image:
        """Convert to RGB, filling transparent pixels with the mean opaque colour.

        PIL's default RGBA→RGB fills alpha=0 areas with black, creating a dark
        silhouette of masked-out content (e.g. a tower hole in a sky crop) that
        causes BLIP to caption the hole instead of the actual content."""
        pil = image.image
        if pil.mode != "RGBA":
            return image.rgb()
        arr = np.array(pil).astype(np.float32)
        alpha = arr[..., 3:4] / 255.0
        rgb = arr[..., :3]
        opaque = arr[..., 3] > 128
        mean_color = rgb[opaque].mean(axis=0) if opaque.any() else np.array([128.0, 128.0, 128.0])
        background = np.ones_like(rgb) * mean_color
        composited = (rgb * alpha + background * (1.0 - alpha)).astype(np.uint8)
        return PILImage.fromarray(composited, mode="RGB")

    def caption(
        self,
        input: Image,
        scene_image: Image | None = None,
        box: list[float] | None = None,
        prompt: str = "",
    ) -> str:
        if scene_image is not None:
            pil_input = make_context_composite(self._to_rgb(input), scene_image.rgb(), box)
        else:
            pil_input = self._to_rgb(input)

        if self.model_type == CaptioningModel.BLIP:
            return self._caption_blip(pil_input, prompt)
        return self._caption_florence(pil_input)

    def _caption_blip(self, pil_input: PILImage.Image, prompt: str) -> str:
        inputs = self.processor(pil_input, prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.net.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def _caption_florence(self, pil_input: PILImage.Image) -> str:
        inputs = self.processor(
            text=self._FLORENCE_TASK,
            images=pil_input,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.net.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )

        raw = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            raw,
            task=self._FLORENCE_TASK,
            image_size=(pil_input.width, pil_input.height),
        )
        return parsed.get(self._FLORENCE_TASK, "").strip()
