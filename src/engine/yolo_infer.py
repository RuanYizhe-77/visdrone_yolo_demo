import numpy as np


VISDRONE_NAMES = [
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

COCO_TO_VISDRONE = {
    0: 0,  # person -> pedestrian
    1: 2,  # bicycle -> bicycle
    2: 3,  # car -> car
    3: 9,  # motorcycle -> motor
    5: 8,  # bus -> bus
    7: 5,  # truck -> truck
}


def empty_np_prediction():
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }


def yolo_label_mode(yolo):
    names = getattr(yolo, "names", None)
    if isinstance(names, dict):
        ordered = [str(names[i]).lower() for i in sorted(names)]
    elif names is not None:
        ordered = [str(name).lower() for name in names]
    else:
        return "unknown"
    if ordered == VISDRONE_NAMES:
        return "visdrone"
    if len(ordered) == 80 and ordered[:4] == ["person", "bicycle", "car", "motorcycle"]:
        return "coco"
    return "unknown"


def predict_yolo(yolo, image_bgr, img_size, conf, device, iou=0.5, max_detections=300):
    if yolo is None:
        return empty_np_prediction()
    result = yolo.predict(
        image_bgr,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        max_det=max_detections,
        device=device,
        verbose=False,
    )[0]
    boxes, scores, labels = [], [], []
    if result.boxes is None:
        return empty_np_prediction()
    label_mode = yolo_label_mode(yolo)
    for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        cls = int(cls)
        if label_mode == "visdrone":
            mapped = cls if 0 <= cls < len(VISDRONE_NAMES) else None
        elif label_mode == "coco":
            mapped = COCO_TO_VISDRONE.get(cls)
        else:
            mapped = cls if 0 <= cls < len(VISDRONE_NAMES) else None
        if mapped is None:
            continue
        boxes.append(box.astype(np.float32))
        scores.append(float(score))
        labels.append(mapped)
    if not boxes:
        return empty_np_prediction()
    return {"boxes": np.asarray(boxes, np.float32), "scores": np.asarray(scores, np.float32), "labels": np.asarray(labels, np.int64)}
