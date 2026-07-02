from enum import Enum


class PanoramaLoraType(Enum):
    """LoRA weights loadable onto the FLUX.1-dev panorama/inpainting pipelines."""

    LAYER_PANO_3D = 1
    FLUX_DEV_PANORAMA_LORA_2 = 2
    FLUX_SEAMLESS_TEXTURE = 3

    @classmethod
    def default(cls):
        return cls.LAYER_PANO_3D


# base_model is only consulted by InPaintingFlux (see lora_base_model), which is used by
# FLUX_DEV_PANORAMA_LORA_2 (skybox inpainting) and FLUX_SEAMLESS_TEXTURE (terrain texture
# generation). Both were trained on base FLUX.1-dev (plain text-to-image), not on
# FLUX.1-Fill-dev — Fill-dev shares the same transformer/attention blocks as dev and only
# differs at the input patch-embedding layer (extra channels for mask/masked-image
# conditioning), so a dev-trained LoRA *can* be cross-loaded onto FluxFillPipeline, but it
# never saw Fill's mask conditioning during training. InPaintingFlux therefore loads each of
# these two onto the base model they were actually trained on, falling back to
# FluxFillPipeline only when no LoRA is requested at all.
#
# LAYER_PANO_3D has no base_model entry: it's only ever used by image_panorama_flux_imp.py,
# which hardcodes FLUX.1-dev directly and never goes through InPaintingFlux.
_LORA_SPECS: dict[PanoramaLoraType, dict[str, str]] = {
    PanoramaLoraType.LAYER_PANO_3D: {
        "checkpoint_dir": "layer_pano_3d",
        "weight_name": "pano_lora_720*1440_v1.safetensors",
        "prompt_prefix": "",
        "prompt_suffix": "",
    },
    PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2: {
        "checkpoint_dir": "flux_panorama_lora",
        "weight_name": "flux_train_replicate.safetensors",
        # jbilcke-hf/flux-dev-panorama-lora-2 was trained with the instance token "TOK",
        # in the form "HDRI panoramic view of TOK, <scene description>".
        "prompt_prefix": "HDRI panoramic view of TOK, ",
        "prompt_suffix": "",
        "base_model": "black-forest-labs/FLUX.1-dev",
    },
    PanoramaLoraType.FLUX_SEAMLESS_TEXTURE: {
        "checkpoint_dir": "flux_seamless_texture",
        "weight_name": "seamless_texture.safetensors",
        # gokaygokay/Flux-Seamless-Texture-LoRA was trained with the instance token
        # "smlstxtr", in the form "smlstxtr, <description>, seamless texture".
        "prompt_prefix": "smlstxtr, ",
        "prompt_suffix": ", seamless texture",
        "base_model": "black-forest-labs/FLUX.1-dev",
    },
}


def lora_checkpoint_dir(lora: PanoramaLoraType) -> str:
    return _LORA_SPECS[lora]["checkpoint_dir"]


def lora_weight_name(lora: PanoramaLoraType) -> str:
    return _LORA_SPECS[lora]["weight_name"]


def lora_prompt_prefix(lora: PanoramaLoraType) -> str:
    return _LORA_SPECS[lora]["prompt_prefix"]


def lora_prompt_suffix(lora: PanoramaLoraType) -> str:
    return _LORA_SPECS[lora]["prompt_suffix"]


def lora_base_model(lora: PanoramaLoraType) -> str:
    """The HuggingFace repo id of the FLUX checkpoint this LoRA was trained on."""
    return _LORA_SPECS[lora]["base_model"]
