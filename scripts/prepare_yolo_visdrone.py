import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visdrone.classes import CATEGORY_TO_CLASS, VALID_CATEGORIES
from src.visdrone.discovery import find_det_split


NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def convert_split(data_root, split):
    _, image_dir, annotation_dir = find_det_split(data_root, split)
    label_dir = image_dir.parent / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    for image_path in tqdm(image_paths, desc=f"convert {split}"):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        rows = []
        ann_path = annotation_dir / f"{image_path.stem}.txt"
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) < 8:
                        continue
                    x, y, w, h = map(float, parts[:4])
                    category = int(float(parts[5]))
                    if category not in VALID_CATEGORIES or w <= 1 or h <= 1:
                        continue
                    x1 = max(0.0, x)
                    y1 = max(0.0, y)
                    x2 = min(float(width - 1), x + w)
                    y2 = min(float(height - 1), y + h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cls = CATEGORY_TO_CLASS[category]
                    xc = ((x1 + x2) * 0.5) / width
                    yc = ((y1 + y2) * 0.5) / height
                    bw = (x2 - x1) / width
                    bh = (y2 - y1) / height
                    rows.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (label_dir / f"{image_path.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return image_dir, label_dir, len(image_paths)


def write_yaml(data_root, output):
    train_dir, _, n_train = convert_split(data_root, "train")
    val_dir, _, n_val = convert_split(data_root, "val")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(NAMES))
    output.write_text(
        f"path: {Path(data_root).resolve()}\n"
        f"train: {train_dir.resolve()}\n"
        f"val: {val_dir.resolve()}\n"
        "nc: 10\n"
        "names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )
    print(f"wrote {output} ({n_train} train images, {n_val} val images)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--output", default="configs/visdrone_yolo.yaml")
    args = p.parse_args()
    write_yaml(args.data_root, args.output)


if __name__ == "__main__":
    main()
