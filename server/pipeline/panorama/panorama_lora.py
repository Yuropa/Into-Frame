from enum import Enum


class PanoramaLoraType(Enum):
    """LoRA weights loadable onto the FLUX.1-dev panorama/inpainting pipelines."""

    LAYER_PANO_3D = 1
    FLUX_DEV_PANORAMA_LORA_2 = 2
    FLUX_SEAMLESS_TEXTURE = 3

    @classmethod
    def default(cls):
        return cls.LAYER_PANO_3D


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
    },
    PanoramaLoraType.FLUX_SEAMLESS_TEXTURE: {
        "checkpoint_dir": "flux_seamless_texture",
        "weight_name": "seamless_texture.safetensors",
        # gokaygokay/Flux-Seamless-Texture-LoRA was trained with the instance token
        # "smlstxtr", in the form "smlstxtr, <description>, seamless texture".
        "prompt_prefix": "smlstxtr, ",
        "prompt_suffix": ", seamless texture",
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
