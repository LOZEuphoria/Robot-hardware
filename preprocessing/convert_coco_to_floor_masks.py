import json
from pathlib import Path
from PIL import Image, ImageDraw


DATA_ROOT = Path("dataset")
SPLITS = ["train", "valid", "test"]

ANNOTATION_NAME = "_annotations.coco.json"

# Roboflow file của bạn có category "Floor-" và "floor"
# Ta chỉ lấy "floor" làm vùng sàn thật.
TARGET_CATEGORY_NAMES = {"floor"}


def find_image_path(image_dir: Path, file_name: str):
    direct_path = image_dir / file_name

    if direct_path.exists():
        return direct_path

    stem = Path(file_name).stem

    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def draw_polygon_on_mask(mask_img, segmentation, value=1):
    """
    COCO polygon segmentation:
    segmentation = [
        [x1, y1, x2, y2, ...],
        ...
    ]
    """

    draw = ImageDraw.Draw(mask_img)

    if not isinstance(segmentation, list):
        return mask_img

    for poly in segmentation:
        if not poly or len(poly) < 6:
            continue

        points = []

        for i in range(0, len(poly), 2):
            x = float(poly[i])
            y = float(poly[i + 1])
            points.append((x, y))

        draw.polygon(points, fill=value)

    return mask_img


def create_preview(image_rgb, mask_img):
    """
    Overlay xanh lên vùng floor để kiểm tra bằng mắt.
    Mask train vẫn là 0/1.
    """

    image_rgb = image_rgb.convert("RGBA")
    mask_img = mask_img.convert("L")

    green_layer = Image.new("RGBA", image_rgb.size, (0, 255, 0, 110))

    # Chỉ overlay ở vùng mask > 0
    alpha_mask = mask_img.point(lambda p: 110 if p > 0 else 0)

    overlay = Image.new("RGBA", image_rgb.size, (0, 0, 0, 0))
    overlay.paste(green_layer, (0, 0), alpha_mask)

    result = Image.alpha_composite(image_rgb, overlay)

    return result.convert("RGB")


def convert_split(split: str):
    image_dir = DATA_ROOT / split
    ann_path = image_dir / ANNOTATION_NAME

    if not image_dir.exists():
        print(f"[SKIP] Không có folder ảnh: {image_dir}")
        return

    if not ann_path.exists():
        print(f"[SKIP] Không có annotation: {ann_path}")
        return

    mask_dir = DATA_ROOT / "masks" / split
    preview_dir = DATA_ROOT / "previews" / split

    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== CONVERT {split.upper()} =====")
    print(f"Images     : {image_dir}")
    print(f"Annotation : {ann_path}")
    print(f"Masks      : {mask_dir}")
    print(f"Previews   : {preview_dir}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])

    target_category_ids = set()

    print("\nCategories:")

    for cat in categories:
        cat_id = int(cat["id"])
        cat_name = cat["name"]

        print(f"  id={cat_id}, name={cat_name}")

        if cat_name in TARGET_CATEGORY_NAMES:
            target_category_ids.add(cat_id)

    if not target_category_ids:
        raise RuntimeError(
            f"Không tìm thấy category trong {TARGET_CATEGORY_NAMES}. "
            "Kiểm tra lại tên category trong COCO json."
        )

    print(f"\nTarget category ids: {target_category_ids}")

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    image_id_to_info = {
        int(img["id"]): img
        for img in images
    }

    image_id_to_annotations = {}

    for ann in annotations:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])

        if category_id not in target_category_ids:
            continue

        image_id_to_annotations.setdefault(image_id, []).append(ann)

    converted_count = 0
    no_annotation_count = 0
    missing_image_count = 0

    for image_id, img_info in image_id_to_info.items():
        file_name = img_info["file_name"]

        image_path = find_image_path(image_dir, file_name)

        if image_path is None:
            print(f"[MISSING IMAGE] {file_name}")
            missing_image_count += 1
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        mask = Image.new("L", (width, height), 0)

        anns = image_id_to_annotations.get(image_id, [])

        if not anns:
            no_annotation_count += 1

        for ann in anns:
            segmentation = ann.get("segmentation", None)

            if segmentation is None:
                continue

            mask = draw_polygon_on_mask(
                mask,
                segmentation,
                value=1
            )

        mask_path = mask_dir / f"{image_path.stem}.png"
        preview_path = preview_dir / f"{image_path.stem}_preview.png"

        # Mask dùng để train: 0/1
        mask.save(mask_path)

        # Preview nhìn bằng mắt: overlay xanh
        preview = create_preview(image, mask)
        preview.save(preview_path)

        converted_count += 1

    print(f"\nConverted images       : {converted_count}")
    print(f"No floor annotation    : {no_annotation_count}")
    print(f"Missing/read fail image: {missing_image_count}")


def main():
    for split in SPLITS:
        convert_split(split)

    print("\nDone.")
    print("Kiểm tra preview trong:")
    print(DATA_ROOT / "previews")


if __name__ == "__main__":
    main()