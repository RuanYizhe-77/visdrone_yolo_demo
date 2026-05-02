from dataclasses import dataclass

import numpy as np


def iou_one(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    cls: int
    score: float
    age: int = 0


class SimpleIoUTracker:
    """Small IoU tracker, inspired by ByteTrack-style online matching but not ByteTrack."""

    def __init__(self, iou_thresh=0.3, max_age=20):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        boxes = np.asarray(detections.get("boxes", []), dtype=np.float32)
        scores = np.asarray(detections.get("scores", []), dtype=np.float32)
        labels = np.asarray(detections.get("labels", []), dtype=np.int64)
        unmatched = set(range(len(boxes)))

        for track in self.tracks:
            track.age += 1
            if len(unmatched) == 0:
                continue
            cand = np.array(sorted(unmatched))
            same_cls = labels[cand] == track.cls
            if not same_cls.any():
                continue
            cand = cand[same_cls]
            ious = iou_one(track.bbox, boxes[cand])
            best_local = int(ious.argmax())
            if ious[best_local] >= self.iou_thresh:
                det_idx = int(cand[best_local])
                track.bbox = boxes[det_idx]
                track.score = float(scores[det_idx])
                track.cls = int(labels[det_idx])
                track.age = 0
                unmatched.remove(det_idx)

        for det_idx in sorted(unmatched):
            self.tracks.append(Track(self.next_id, boxes[det_idx], int(labels[det_idx]), float(scores[det_idx]), age=0))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [
            {"track_id": t.track_id, "bbox": t.bbox.copy(), "class_id": t.cls, "score": t.score, "age": t.age}
            for t in self.tracks
            if t.age == 0
        ]
