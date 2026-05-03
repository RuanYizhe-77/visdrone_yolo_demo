from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .classes import CATEGORY_TO_CLASS, VALID_CATEGORIES
from .discovery import find_det_split


def read_det_annotation(path, width, height):
    boxes, labels = [], []
    if not Path(path).exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 8:
                continue
            x, y, w, h = map(float, parts[:4])
            category = int(float(parts[5]))
            if category not in VALID_CATEGORIES or w <= 1 or h <= 1:
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width - 1), x + w), min(float(height - 1), y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CATEGORY_TO_CLASS[category])
    return np.asarray(boxes, np.float32), np.asarray(labels, np.int64)


class VisDroneDETDataset(Dataset):
    def __init__(self, data_root=".", split="train", img_size=640, max_samples=None, augment=False):
        _, self.image_dir, self.annotation_dir = find_det_split(data_root, split)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in exts)
        if max_samples:
            self.images = self.images[: int(max_samples)]
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]
        boxes, labels = read_det_annotation(self.annotation_dir / f"{path.stem}.txt", w0, h0)

        if self.augment and len(boxes) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = w0 - 1 - x2
            boxes[:, 2] = w0 - 1 - x1

        sx, sy = self.img_size / w0, self.img_size / h0
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "orig_size": torch.tensor([h0, w0], dtype=torch.long),
            "image_size": torch.tensor([self.img_size, self.img_size], dtype=torch.long),
            "path": str(path),
        }
        return tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

