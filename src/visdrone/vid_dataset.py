from pathlib import Path

import cv2
import numpy as np

from .classes import CATEGORY_TO_CLASS, VALID_CATEGORIES
from .discovery import find_vid_split


def read_vid_annotation(path):
    by_frame = {}
    if not Path(path).exists():
        return by_frame
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 10:
                continue
            frame_id = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            category = int(float(parts[7]))
            if category not in VALID_CATEGORIES or w <= 1 or h <= 1:
                continue
            by_frame.setdefault(frame_id, []).append(
                {
                    "track_id": obj_id,
                    "bbox": [x, y, x + w, y + h],
                    "class_id": CATEGORY_TO_CLASS[category],
                }
            )
    return by_frame


class VisDroneVIDSequence:
    def __init__(self, sequence_dir, annotation_path=None):
        self.sequence_dir = Path(sequence_dir)
        self.frames = sorted(
            p for p in self.sequence_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.annotations = read_vid_annotation(self.annotation_path) if self.annotation_path else {}
        if not self.frames:
            raise FileNotFoundError(f"No frames found in {self.sequence_dir}")

    @property
    def name(self):
        return self.sequence_dir.name

    def __len__(self):
        return len(self.frames)

    def iter_frames(self):
        for idx, path in enumerate(self.frames, start=1):
            image = cv2.imread(str(path))
            if image is None:
                continue
            yield idx, path, image, self.annotations.get(idx, [])


def list_vid_sequences(data_root=".", split="val", max_sequences=None):
    _, seq_root, ann_root = find_vid_split(data_root, split)
    seqs = []
    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        ann = ann_root / f"{seq_dir.name}.txt"
        seqs.append(VisDroneVIDSequence(seq_dir, ann if ann.exists() else None))
        if max_sequences and len(seqs) >= int(max_sequences):
            break
    return seqs

