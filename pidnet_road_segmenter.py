import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PIDNetRoadSegmenter:
    def __init__(
        self,
        pidnet_root,
        weight_path,
        model_size="pidnet-s",
        num_classes=19,
        device=None,
        seg_width=320,
        seg_height=240
    ):
        self.pidnet_root = Path(pidnet_root)
        self.weight_path = Path(weight_path)
        self.model_size = model_size
        self.num_classes = num_classes
        self.seg_width = seg_width
        self.seg_height = seg_height

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).lower() == "directml":
            try:
                import torch_directml
                self.device = torch_directml.device()
            except ImportError as exc:
                raise RuntimeError(
                    "Chua cai torch-directml trong moi truong camera"
                ) from exc
        else:
            self.device = device

        if not self.pidnet_root.exists():
            raise FileNotFoundError(f"Không thấy PIDNet root: {self.pidnet_root}")

        if not self.weight_path.exists():
            raise FileNotFoundError(f"Không thấy weight PIDNet: {self.weight_path}")

        sys.path.insert(0, str(self.pidnet_root))

        import models

        self.model = models.pidnet.get_pred_model(
            self.model_size,
            self.num_classes
        )

        self.model = self._load_pretrained(self.model, self.weight_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"PIDNet loaded on device: {self.device}")
        print(f"Model: {self.model_size}")
        print(f"Weight: {self.weight_path}")

    def _load_pretrained(self, model, pretrained):
        checkpoint = torch.load(pretrained, map_location="cpu")

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

        model_dict = model.state_dict()

        loaded_dict = {}

        for k, v in checkpoint.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[7:]
            if new_key.startswith("model."):
                new_key = new_key[6:]

            if new_key in model_dict and v.shape == model_dict[new_key].shape:
                loaded_dict[new_key] = v

        print(f"Loaded PIDNet parameters: {len(loaded_dict)}")

        model_dict.update(loaded_dict)
        model.load_state_dict(model_dict, strict=False)

        return model

    def _preprocess(self, frame_bgr):
        small = cv2.resize(
            frame_bgr,
            (self.seg_width, self.seg_height),
            interpolation=cv2.INTER_AREA
        )

        # BGR -> RGB
        img = small[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - MEAN) / STD

        img = img.transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        return tensor

    def predict_seg_map(self, frame_bgr):
        original_h, original_w = frame_bgr.shape[:2]

        tensor = self._preprocess(frame_bgr)

        with torch.no_grad():
            pred = self.model(tensor)

            if isinstance(pred, (list, tuple)):
                pred = pred[0]
                print(pred)
            pred = F.interpolate(
                pred,
                size=(self.seg_height, self.seg_width),
                mode="bilinear",
                align_corners=True
            )

            seg_small = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)

        seg_map = cv2.resize(
            seg_small,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        return seg_map

    def create_road_mask(
        self,
        seg_map,
        road_class_id=0,
        sidewalk_class_id=1,
        use_sidewalk=False
    ):
        road_mask = seg_map == road_class_id

        if use_sidewalk:
            road_mask = road_mask | (seg_map == sidewalk_class_id)

        return road_mask.astype(np.uint8) * 255

    def overlay_road(self, frame_bgr, road_mask, alpha=0.45):
        result = frame_bgr.copy()

        color_layer = np.zeros_like(result, dtype=np.uint8)
        color_layer[:, :] = (0, 255, 0)

        mask_bool = road_mask > 0

        blended = cv2.addWeighted(
            result,
            1.0 - alpha,
            color_layer,
            alpha,
            0
        )

        result[mask_bool] = blended[mask_bool]

        return result
