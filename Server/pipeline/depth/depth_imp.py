from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from util.json_utils import write_json
from depth_anything_3.api import DepthAnything3
from remote_connection.remote_server import RemoteServer

class DepthGenerator(RemoteServer):
    def __init__(self) -> None:
        super().__init__()

        self.model = DepthAnything3.from_pretrained(
            "depth-anything/da3nested-giant-large"
        )
        self.model = self.model.to(device=self.device)
    
    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "depth":
            return self.depth(temp_path, input)
        raise ValueError(f"Unknown action: {action}")
    
    def depth(self, temp_path: Path, input: Image) -> Any:
        image = self.decode_image(input).convert("RGB")
        with torch.no_grad():
            result = self.model.inference([image])

        print(f"extrinsics {result.extrinsics} {type(result.extrinsics)}")
        return result
    
if __name__ == "__main__":
    DepthGenerator.run()