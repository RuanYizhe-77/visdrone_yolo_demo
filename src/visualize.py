import cv2

from .utils import VISDRONE_CLASSES


def draw_detections(image, boxes, scores, labels, track_ids=None):
    out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls = int(labels[i])
        score = float(scores[i])
        color = class_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = VISDRONE_CLASSES[cls] if 0 <= cls < len(VISDRONE_CLASSES) else str(cls)
        prefix = f"ID {track_ids[i]} " if track_ids is not None else ""
        text = f"{prefix}{name} {score:.2f}"
        cv2.putText(out, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def class_color(cls):
    palette = [
        (220, 20, 60), (255, 140, 0), (0, 128, 255), (60, 179, 113), (147, 112, 219),
        (0, 191, 255), (255, 215, 0), (199, 21, 133), (70, 130, 180), (46, 139, 87),
    ]
    return palette[int(cls) % len(palette)]
