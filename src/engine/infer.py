import time
from pathlib import Path

import cv2
import torch

from src.modeling.decode import decode_fcos
from src.visualization.draw import draw_boxes


def preprocess_bgr(image_bgr, img_size, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device), (h0, w0)


@torch.no_grad()
def predict_image(
    model,
    image_bgr,
    img_size=640,
    device="cuda",
    conf=0.25,
    iou=0.5,
    max_detections=200,
    nms_mode="classwise",
    agnostic_nms_thresh=0.7,
):
    model.eval()
    x, (h0, w0) = preprocess_bgr(image_bgr, img_size, device)
    start = time.perf_counter()
    outputs = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - start) * 1000.0
    pred = decode_fcos(
        outputs,
        model.strides,
        score_thresh=conf,
        nms_thresh=iou,
        max_detections=max_detections,
        nms_mode=nms_mode,
        agnostic_nms_thresh=agnostic_nms_thresh,
        regress_normalized=getattr(model, "regress_normalized", False),
    )[0]
    sx, sy = w0 / img_size, h0 / img_size
    boxes = pred["boxes"].detach().cpu()
    if len(boxes):
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
    return {
        "boxes": boxes,
        "scores": pred["scores"].detach().cpu(),
        "labels": pred["labels"].detach().cpu(),
        "infer_ms": infer_ms,
    }


def save_prediction_image(
    model,
    image_path,
    output_dir,
    img_size=640,
    device="cuda",
    conf=0.25,
    iou=0.5,
    max_detections=200,
    nms_mode="classwise",
    agnostic_nms_thresh=0.7,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    pred = predict_image(
        model,
        image,
        img_size,
        device,
        conf,
        iou,
        max_detections=max_detections,
        nms_mode=nms_mode,
        agnostic_nms_thresh=agnostic_nms_thresh,
    )
    vis = draw_boxes(image, pred["boxes"].numpy(), pred["labels"].numpy(), pred["scores"].numpy())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(image_path).name
    cv2.imwrite(str(out_path), vis)
    return out_path, pred
