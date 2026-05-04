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
    _draw_panel_title(canvas, left_title, 0, left.shape[1])
    _draw_panel_title(canvas, right_title, left.shape[1], right.shape[1])
    return canvas


def _draw_panel_title(canvas, title, x, panel_width):
    label = title.replace("_", " ")
    cv2.rectangle(canvas, (x, 0), (x + panel_width, 44), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.78
    thickness = 2
    (tw, _), _ = cv2.getTextSize(label, font, scale, thickness)
    if tw > panel_width - 24:
        scale = max(0.48, scale * (panel_width - 24) / max(tw, 1))
    cv2.putText(canvas, label, (x + 12, 30), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
