import torch
import numpy as np
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from util.image_utils import Image
from pipeline.segmentation.segmentation_result import SegmentationResult

class ForegroundSegmentationResult():
    def __init__(self, mask, image) -> None:
        self.mask = Image(mask)
        self.image = Image(image)

class ForegroundSeg:
    def __init__(self, device):
        self.device = device

        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True
        )

        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.birefnet.to(device)
        self.birefnet.half()
        self.birefnet.eval()

        image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @classmethod
    def model_names(cls) -> list[str]:
        return ['ZhengPeng7/BiRefNet']

    def segment(self, input: Image) -> SegmentationResult:
        image = input.image.convert("RGB")
        input_images = self.transform_image(image).unsqueeze(0).to(self.device).half()

        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)

        # Convert mask to boolean numpy array
        mask_np = np.array(mask) > 127  # (H, W) bool

        # Compute bounding box from mask
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)

        if rows.any() and cols.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            bbox = {
                'x': int(x_min),
                'y': int(y_min),
                'w': int(x_max - x_min),
                'h': int(y_max - y_min)
            }
            score = float(np.array(mask).mean() / 255.0)  # rough confidence proxy
        else:
            # Empty mask fallback
            bbox = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
            score = 0.0

        return SegmentationResult(mask_np, bbox, score)
