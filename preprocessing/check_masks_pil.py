from pathlib import Path
from PIL import Image


DATA_ROOT = Path("dataset")

for split in ["train", "valid", "test"]:
    mask_dir = DATA_ROOT / "masks" / split

    print(f"\nChecking {split}: {mask_dir}")

    paths = sorted(mask_dir.glob("*.png"))

    if not paths:
        print("No masks found.")
        continue

    for path in paths[:20]:
        mask = Image.open(path).convert("L")
        values = sorted(set(mask.getdata()))
        print(path.name, values[:10])