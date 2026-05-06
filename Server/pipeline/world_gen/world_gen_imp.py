from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from util.json_utils import write_json
from worldgen import WorldGen
from remote_connection.remote_server import RemoteServer

class WorldGenerator(RemoteServer):
    def setup(self):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.worldgen = WorldGen(mode="i2s", device=self.device, low_vram=True)
    
    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "generate":
            return self.generate(temp_path, input)
        raise ValueError(f"Unknown action: {action}")
    
    def _resize_max(self, img, max_size=768):
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        return img

    def generate(self, temp_path: Path, input: Image) -> Any:
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                image = self._resize_max(input)
                result = self.worldgen.generate_pano(image=input)

        return result
    
if __name__ == "__main__":
    WorldGenerator.run()