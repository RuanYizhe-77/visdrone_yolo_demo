from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_VISDRONE_CATEGORIES = set(range(1, 11))


def discover_visdrone(data_root=".", split="train"):
    root = Path(data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"data-root does not exist: {root}")

    official = f"VisDrone2019-DET-{split}"
    for p in root.rglob(official):
        if (p / "images").is_dir() and (p / "annotations").is_dir():
            return p / "images", p / "annotations", "visdrone"

    # Fallback for repos converted to image/label split folders.
    for p in root.rglob("images"):
        img_dir = p / split
        label_dir = p.parent / "labels" / split
        if img_dir.is_dir() and label_dir.is_dir():
            return img_dir, label_dir, "yolo"

    msg = (
        f"Could not find VisDrone {split} data under {root}.\n"
        f"Expected either {official}/images + {official}/annotations, "
        f"or a converted images/{split} + labels/{split} layout."
    )
    raise FileNotFoundError(msg)


class VisDroneDetectionDataset(Dataset):
    def __init__(self, data_root=".", split="train", img_size=512, max_samples=None):
        self.img_dir, self.ann_dir, self.format = discover_visdrone(data_root, split)
        self.img_size = int(img_size)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in exts)
        if max_samples:
            self.images = self.images[: int(max_samples)]
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]
        boxes, labels = self.read_annotations(img_path, w0, h0)

        sx = self.img_size / w0
        sy = self.img_size / h0
        resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        targets = build_targets(boxes, labels, self.img_size, stride=4, num_classes=10)
        raw_target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32), "labels": torch.as_tensor(labels, dtype=torch.long)}
        return img_tensor, targets, raw_target, str(img_path)

    def read_annotations(self, img_path, width, height):
        ann_path = self.ann_dir / f"{img_path.stem}.txt"
        boxes, labels = [], []
        if not ann_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                vals = [v.strip() for v in line.strip().split(",") if v.strip()]
                if len(vals) < 5:
                    vals = line.strip().split()
                if self.format == "visdrone" and len(vals) >= 6:
                    x, y, bw, bh = map(float, vals[:4])
                    cat = int(float(vals[5]))
                    if cat not in VALID_VISDRONE_CATEGORIES or bw <= 1 or bh <= 1:
                        continue
                    boxes.append([x, y, x + bw, y + bh])
                    labels.append(cat - 1)
                elif self.format == "yolo" and len(vals) >= 5:
                    cls = int(float(vals[0]))
                    cx, cy, bw, bh = map(float, vals[1:5])
                    x1 = (cx - bw / 2) * width
                    y1 = (cy - bh / 2) * height
                    x2 = (cx + bw / 2) * width
                    y2 = (cy + bh / 2) * height
                    if 0 <= cls < 10 and x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)
        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
        return boxes, labels


def build_targets(boxes, labels, img_size, stride=4, num_classes=10):
    out = img_size // stride
    heatmap = torch.zeros((num_classes, out, out), dtype=torch.float32)
    wh = torch.zeros((2, out, out), dtype=torch.float32)
    offset = torch.zeros((2, out, out), dtype=torch.float32)
    mask = torch.zeros((out, out), dtype=torch.float32)

    for box, cls in zip(boxes, labels):
        x1, y1, x2, y2 = box / stride
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ix, iy = int(cx), int(cy)
        if ix < 0 or iy < 0 or ix >= out or iy >= out:
            continue
        heatmap[int(cls), iy, ix] = 1.0
        wh[:, iy, ix] = torch.tensor([x2 - x1, y2 - y1])
        offset[:, iy, ix] = torch.tensor([cx - ix, cy - iy])
        mask[iy, ix] = 1.0
    return {"heatmap": heatmap, "wh": wh, "offset": offset, "mask": mask}


def collate_fn(batch):
    imgs, targets, raw_targets, paths = zip(*batch)
    imgs = torch.stack(imgs)
    out_targets = {k: torch.stack([t[k] for t in targets]) for k in targets[0]}
    return imgs, out_targets, list(raw_targets), list(paths)
