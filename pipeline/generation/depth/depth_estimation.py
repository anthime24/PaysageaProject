import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

class DepthEstimator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def predict(self, image):
        input_img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.model(input_img)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        return depth.astype(np.uint8)
