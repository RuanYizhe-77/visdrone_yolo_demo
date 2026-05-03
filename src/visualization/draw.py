import cv2
import numpy as np

from src.visdrone.classes import VISDRONE_CLASSES

COLORS = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
    ],
    dtype=np.uint8,
)


def draw_boxes(image_bgr, boxes, labels, scores=None, track_ids=None, thickness=2):
    image = image_bgr.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls = int(labels[i])
        color = COLORS[cls % len(COLORS)].tolist()
        text = VISDRONE_CLASSES[cls] if 0 <= cls < len(VISDRONE_CLASSES) else str(cls)
        if scores is not None:
            text += f" {float(scores[i]):.2f}"
        if track_ids is not None:
            text = f"ID {int(track_ids[i])} " + text
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return image


def make_side_by_side(left, right, left_title="ours", right_title="baseline"):
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    cv2.putText(canvas, left_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, right_title, (left.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return canvas

