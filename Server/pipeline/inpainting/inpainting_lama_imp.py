from path_utils import add_project_paths, add_system_path, lib_path, checkpoints_path
add_project_paths()

add_system_path(lib_path() / "LaMa")

import numpy as np
import yaml
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image as PILImage
from omegaconf import OmegaConf

from remote_connection.remote_server import RemoteServer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class InPaintingLama(RemoteServer):
    def setup(self):
        # Add lama to path so saicinpainting is importable
        from saicinpainting.training.trainers import load_checkpoint

        model_path = checkpoints_path() / "lama/big-lama"
        train_config_path = model_path / "config.yaml"
        checkpoint_path = model_path / "models" / "best.ckpt"

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        model = load_checkpoint(train_config, str(checkpoint_path), strict=False, map_location='cpu')
        model.freeze()
        model.to(self.device)
        model.eval()
        return model

    def inpaint(
        self,
        input_image: PILImage.Image,
        mask_image: PILImage.Image,
        temp_path: Path,
        prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        strength: float = 1.0,
    ) -> PILImage.Image:
        from saicinpainting.evaluation.utils import move_to_device

        # Convert image to float32 tensor (H, W, C) -> (C, H, W) in [0, 1]
        image_np = np.array(input_image.convert("RGB")).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

        # Convert mask: white=inpaint, black=keep -> float32 (1, H, W) in [0, 1]
        mask_np = np.array(mask_image.convert("L")).astype(np.float32)
        if mask_np.max() <= 1.0:
            mask_np = mask_np * 255.0
        mask_np = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        batch = {
            'image': image_tensor,
            'mask':  mask_tensor,
        }
        batch = default_collate([{k: v.squeeze(0) for k, v in batch.items()}])

        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = self.model(batch)
            result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return PILImage.fromarray(result, mode="RGB")

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "inpaint":
            print(f"Got input: {input}")
            image = input["image"]
            mask = input["mask"]
            prompt = input.get("prompt", "")
            num_inference_steps = input.get("num_inference_steps", 30)
            guidance_scale = input.get("guidance_scale", 1.0)
            strength = input.get("strength", 1.0)

            return self.inpaint(
                input_image=image,
                mask_image=mask,
                temp_path=temp_path,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            )
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    InPaintingLama.run()