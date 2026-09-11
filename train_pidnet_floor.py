import sys
import random
import cv2
import numpy as np
from pathlib import Path

from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

PIDNET_ROOT = Path("PIDNet")

# Optional pretrained Cityscapes weight. Nếu chưa có file này, script sẽ train từ đầu.
PRETRAINED_CITYSCAPES = (
    PIDNET_ROOT
    / "pretrained_models"
    / "cityscapes"
    / "PIDNet_L_Cityscapes_test.pt"
)

DATA_ROOT = Path("dataset")

TRAIN_IMAGE_DIR = DATA_ROOT / "train"
TRAIN_MASK_DIR = DATA_ROOT / "masks" / "train"

VAL_IMAGE_DIR = DATA_ROOT / "valid"
VAL_MASK_DIR = DATA_ROOT / "masks" / "valid"

SAVE_DIR = Path("checkpoints_floor")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# PIDNet binary segmentation:
# 0 = not_floor
# 1 = floor
NUM_CLASSES = 2

MODEL_SIZE = "pidnet-l"

# width, height
# Nếu máy yếu/CPU: (320, 240)
# Nếu có GPU ổn: (512, 384)
IMAGE_SIZE = (512, 384)

BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Windows nên để 0 để tránh lỗi multiprocessing
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Nếu mask của bạn là 0/1 thì để True.
# Nếu mask là 0/255 thì script cũng tự convert mọi pixel > 0 thành floor.
CONVERT_MASK_POSITIVE_TO_FLOOR = True

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# PIL -> TORCH UTILS
# Không dùng cv2/numpy để tránh lỗi import cv2/numpy.
# =========================================================

def pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL RGB image to float tensor [C, H, W], normalized ImageNet.
    Không dùng numpy.
    """
    image = image.convert("RGB")
    w, h = image.size

    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(h, w, 3).permute(2, 0, 1).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    data = (data - mean) / std

    return data


def pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """
    Convert PIL grayscale mask to long tensor [H, W].
    Expected:
      0 = not_floor
      1 = floor
    Nếu mask là 0/255, tự convert >0 thành 1.
    """
    mask = mask.convert("L")
    w, h = mask.size

    data = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    data = data.view(h, w).long()

    if CONVERT_MASK_POSITIVE_TO_FLOOR:
        data = (data > 0).long()

    return data


def generate_edge_map(mask_np, edge_size=4):
    """
    Sinh bản đồ biên từ mask phân vùng mặt sàn (0 và 1) bằng Canny edge detector.
    Theo đúng nguyên bản mã nguồn của tác giả PIDNet.
    """
    mask_255 = (mask_np * 255).astype(np.uint8)
    edge = cv2.Canny(mask_255, 10, 100)
    
    kernel = np.ones((edge_size, edge_size), np.uint8)
    edge = (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)
    return edge


def weighted_bce(bd_pre, target):
    """
    Binary Cross Entropy có trọng số để phạt các pixel biên bị đoán lệch.
    """
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    if sum_num == 0:
        return torch.tensor(0.0, device=bd_pre.device)

    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
    return loss


# =========================================================
# DATASET
# =========================================================

class FloorDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size=(512, 384),
        augment=False
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment

        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.image_paths.extend(self.image_dir.glob(ext))

        self.image_paths = sorted(self.image_paths)

        if not self.image_paths:
            raise RuntimeError(f"Không có ảnh trong: {self.image_dir}")

        # Chỉ giữ ảnh có mask tương ứng
        valid_paths = []
        missing_masks = []

        for img_path in self.image_paths:
            mask_path = self.mask_dir / f"{img_path.stem}.png"

            if mask_path.exists():
                valid_paths.append(img_path)
            else:
                missing_masks.append(img_path.name)

        self.image_paths = valid_paths

        if missing_masks:
            print(f"[WARN] {len(missing_masks)} ảnh thiếu mask trong {self.mask_dir}")
            print("Ví dụ thiếu:", missing_masks[:5])

        if not self.image_paths:
            raise RuntimeError(
                f"Không còn ảnh hợp lệ vì không tìm thấy mask trong: {self.mask_dir}"
            )

        print(
            f"Loaded dataset: images={len(self.image_paths)}, "
            f"image_dir={self.image_dir}, mask_dir={self.mask_dir}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Random brightness
            if random.random() < 0.35:
                factor = random.uniform(0.75, 1.25)
                image = ImageEnhance.Brightness(image).enhance(factor)

            # Random contrast
            if random.random() < 0.35:
                factor = random.uniform(0.80, 1.20)
                image = ImageEnhance.Contrast(image).enhance(factor)

        # Convert mask to numpy for generating edge map
        mask_np = np.array(mask)
        if CONVERT_MASK_POSITIVE_TO_FLOOR:
            mask_np = (mask_np > 0).astype(np.uint8)
            
        edge_np = generate_edge_map(mask_np, edge_size=4)

        image_tensor = pil_rgb_to_tensor(image)
        mask_tensor = torch.from_numpy(mask_np).long()
        edge_tensor = torch.from_numpy(edge_np).float()

        return image_tensor, mask_tensor, edge_tensor


# =========================================================
# MODEL
# =========================================================

def build_pidnet_model(model_size="pidnet-l", num_classes=2):
    """
    Build PIDNet model based on size (pidnet-s, pidnet-m, pidnet-l).

    Cần clone repo:
      git clone https://github.com/XuJiacong/PIDNet.git stereo_vision/PIDNet
    """
    if not PIDNET_ROOT.exists():
        raise FileNotFoundError(
            f"Không thấy PIDNet repo tại: {PIDNET_ROOT}\n"
            f"Hãy chạy: git clone https://github.com/XuJiacong/PIDNet.git {PIDNET_ROOT}"
        )

    sys.path.insert(0, str(PIDNET_ROOT))

    import models

    if model_size == "pidnet-s":
        m, n, planes, ppm_planes, head_planes = 2, 3, 32, 96, 128
    elif model_size == "pidnet-m":
        m, n, planes, ppm_planes, head_planes = 2, 3, 64, 96, 128
    else: # pidnet-l
        m, n, planes, ppm_planes, head_planes = 3, 4, 64, 112, 256

    # Ưu tiên constructor training nếu có.
    if hasattr(models, "pidnet") and hasattr(models.pidnet, "PIDNet"):
        model = models.pidnet.PIDNet(
            m=m,
            n=n,
            num_classes=num_classes,
            planes=planes,
            ppm_planes=ppm_planes,
            head_planes=head_planes,
            augment=True
        )
        return model

    # Fallback: prediction model, vẫn train được main output.
    if hasattr(models, "pidnet") and hasattr(models.pidnet, "get_pred_model"):
        model = models.pidnet.get_pred_model(model_size, num_classes)
        return model

    raise RuntimeError("Không tìm thấy PIDNet builder trong repo models.pidnet")


def load_partial_pretrained(model, weight_path: Path):
    """
    Load partial pretrained Cityscapes.
    Vì pretrained Cityscapes có 19 classes, head cuối không khớp với 2 classes,
    script chỉ load layer nào cùng tên và cùng shape.
    """
    if not weight_path.exists():
        print(f"[INFO] Không thấy pretrained weight: {weight_path}")
        print("[INFO] Sẽ train từ đầu.")
        return model

    print(f"Loading partial pretrained: {weight_path}")

    ckpt = torch.load(str(weight_path), map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    model_dict = model.state_dict()
    loaded = {}

    for k, v in ckpt.items():
        key = k
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("model."):
            key = key[6:]

        if key in model_dict and model_dict[key].shape == v.shape:
            loaded[key] = v

    model_dict.update(loaded)
    model.load_state_dict(model_dict, strict=False)

    print(f"Loaded params: {len(loaded)} / {len(model_dict)}")

    return model


def get_main_output(outputs):
    """
    PIDNet training mode thường trả về list/tuple:
      [detail_output, main_output, boundary_output]
    Ta lấy main_output để train floor segmentation.
    """
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 2:
            return outputs[1]
        return outputs[0]

    return outputs


# =========================================================
# LOSS + METRIC
# =========================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)[:, 1]  # floor probability
        target_floor = (target == 1).float()

        intersection = (probs * target_floor).sum()
        union = probs.sum() + target_floor.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


def compute_floor_iou(logits, target):
    pred = torch.argmax(logits, dim=1)

    pred_floor = pred == 1
    true_floor = target == 1

    intersection = (pred_floor & true_floor).sum().item()
    union = (pred_floor | true_floor).sum().item()

    if union == 0:
        return 1.0

    return intersection / union


# =========================================================
# TRAIN / VAL
# =========================================================

def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss, epoch):
    model.train()

    total_loss = 0.0
    total_iou = 0.0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)

    for images, masks, edges in pbar:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        edges = edges.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        
        # Interpolate outputs to match ground truth size
        target_size = masks.shape[-2:]
        logits_aux = F.interpolate(outputs[0], size=target_size, mode="bilinear", align_corners=False)
        logits_main = F.interpolate(outputs[1], size=target_size, mode="bilinear", align_corners=False)
        logits_bd = F.interpolate(outputs[2], size=target_size, mode="bilinear", align_corners=False)

        # 1. Semantic losses (CE + Dice)
        loss_sem_aux = ce_loss(logits_aux, masks) + dice_loss(logits_aux, masks)
        loss_sem_main = ce_loss(logits_main, masks) + dice_loss(logits_main, masks)
        
        # 2. Boundary loss (weighted BCE)
        loss_boundary = weighted_bce(logits_bd, edges)
        
        # 3. Semantic boundary loss (focus only on pixels with high predicted boundary probability)
        filler = torch.ones_like(masks) * -100
        bd_label = torch.where(torch.sigmoid(logits_bd[:, 0, :, :]) > 0.8, masks, filler)
        loss_sb = ce_loss(logits_main, bd_label)
        
        # Combine losses following paper coefficients: 0.4 * aux + 1.0 * main + 20.0 * bd + 1.0 * sb
        loss = 0.4 * loss_sem_aux + 1.0 * loss_sem_main + 20.0 * loss_boundary + 1.0 * loss_sb

        loss.backward()
        optimizer.step()

        iou = compute_floor_iou(logits_main.detach(), masks)

        total_loss += loss.item()
        total_iou += iou

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "iou": f"{iou:.4f}",
        })

    return total_loss / len(loader), total_iou / len(loader)


@torch.no_grad()
def validate(model, loader, ce_loss, dice_loss):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0

    pbar = tqdm(loader, desc="Validate", leave=False)

    for images, masks, edges in pbar:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        edges = edges.to(DEVICE, non_blocking=True)

        outputs = model(images)
        
        target_size = masks.shape[-2:]
        logits_aux = F.interpolate(outputs[0], size=target_size, mode="bilinear", align_corners=False)
        logits_main = F.interpolate(outputs[1], size=target_size, mode="bilinear", align_corners=False)
        logits_bd = F.interpolate(outputs[2], size=target_size, mode="bilinear", align_corners=False)

        # 1. Semantic losses (CE + Dice)
        loss_sem_aux = ce_loss(logits_aux, masks) + dice_loss(logits_aux, masks)
        loss_sem_main = ce_loss(logits_main, masks) + dice_loss(logits_main, masks)
        
        # 2. Boundary loss (weighted BCE)
        loss_boundary = weighted_bce(logits_bd, edges)
        
        # 3. Semantic boundary loss
        filler = torch.ones_like(masks) * -100
        bd_label = torch.where(torch.sigmoid(logits_bd[:, 0, :, :]) > 0.8, masks, filler)
        loss_sb = ce_loss(logits_main, bd_label)
        
        loss = 0.4 * loss_sem_aux + 1.0 * loss_sem_main + 20.0 * loss_boundary + 1.0 * loss_sb
        iou = compute_floor_iou(logits_main, masks)

        total_loss += loss.item()
        total_iou += iou

    return total_loss / len(loader), total_iou / len(loader)


def save_checkpoint(path, model, optimizer, epoch, val_iou):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_iou": val_iou,
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
        },
        path
    )


# =========================================================
# MAIN
# =========================================================

def main():
    set_seed(SEED)

    print("===== PIDNet Floor Training =====")
    print("DEVICE:", DEVICE)
    print("DATA_ROOT:", DATA_ROOT)
    print("IMAGE_SIZE:", IMAGE_SIZE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("EPOCHS:", EPOCHS)

    train_dataset = FloorDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        image_size=IMAGE_SIZE,
        augment=True
    )

    val_dataset = FloorDataset(
        VAL_IMAGE_DIR,
        VAL_MASK_DIR,
        image_size=IMAGE_SIZE,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )

    model = build_pidnet_model(model_size=MODEL_SIZE, num_classes=NUM_CLASSES)
    model = load_partial_pretrained(model, PRETRAINED_CITYSCAPES)
    model.to(DEVICE)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_iou = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

        train_loss, train_iou = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ce_loss,
            dice_loss,
            epoch
        )

        val_loss, val_iou = validate(
            model,
            val_loader,
            ce_loss,
            dice_loss
        )

        print(
            f"Train loss={train_loss:.4f}, IoU={train_iou:.4f} | "
            f"Val loss={val_loss:.4f}, IoU={val_iou:.4f}"
        )

        last_path = SAVE_DIR / "pidnet_floor_last.pt"
        save_checkpoint(last_path, model, optimizer, epoch, val_iou)

        if val_iou > best_iou:
            best_iou = val_iou

            best_path = SAVE_DIR / "pidnet_floor_best.pt"
            save_checkpoint(best_path, model, optimizer, epoch, val_iou)

            print(f"Saved best checkpoint: {best_path} | Best IoU={best_iou:.4f}")

    print("\nTraining done.")
    print(f"Best val IoU: {best_iou:.4f}")
    print("Best checkpoint:")
    print(SAVE_DIR / "pidnet_floor_best.pt")


if __name__ == "__main__":
    main()
