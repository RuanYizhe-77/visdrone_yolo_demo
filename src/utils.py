import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import cv2


VISDRONE_CLASSES = [
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


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_checkpoint(model, weights, device):
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    return ckpt


def save_checkpoint(path, model, optimizer, epoch, metrics=None):
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics or {},
        },
        path,
    )


def image_files(source):
    source = Path(source)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    raise FileNotFoundError(f"Source not found: {source}")


def preprocess_bgr(image_bgr, img_size, device):
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device), (w / img_size, h / img_size)


def scale_boxes_to_original(boxes, scale):
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale[0]
    boxes[:, [1, 3]] *= scale[1]
    return boxes


def repo_root():
    return Path(os.getcwd()).resolve()
