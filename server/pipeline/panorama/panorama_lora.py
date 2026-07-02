from enum import Enum


class PanoramaLoraType(Enum):
    """LoRA weights loadable onto the FLUX.1-dev panorama pipelines."""

    LAYER_PANO_3D = 1
    FLUX_DEV_PANORAMA_LORA_2 = 2

    @classmethod
    def default(cls):
        return cls.LAYER_PANO_3D


_LORA_SPECS: dict[PanoramaLoraType, dict[str, str]] = {
    PanoramaLoraType.LAYER_PANO_3D: {
        "checkpoint_dir": "layer_pano_3d",
        "weight_name": "pano_lora_720*1440_v1.safetensors",
        "prompt_prefix": "",
    },
    PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2: {
        "checkpoint_dir": "flux_panorama_lora",
        "weight_name": "flux_train_replicate.safetensors",
        # jbilcke-hf/flux-dev-panorama-lora-2 was trained with the instance token "TOK",
        # in the form "HDRI panoramic view of TOK, <scene description>".
        "prompt_prefix": "HDRI panoramic view of TOK, ",
    },
}


def lora_checkpoint_dir(lora: PanoramaLoraType) -> str:
    return _LORA_SPECS[lora]["checkpoint_dir"]


def lora_weight_name(lora: PanoramaLoraType) -> str:
    return _LORA_SPECS[lora]["weight_name"]


def lora_prompt_prefix(lora: PanoramaLoraType) -> str:
    return _LORA_SPECS[lora]["prompt_prefix"]
