import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.modeling.decode import decode_fcos
from src.common.metrics import evaluate_map50


@torch.no_grad()
def evaluate_detector(
    model,
    loader,
    device="cuda",
    conf=0.05,
    iou=0.5,
    output_json=None,
    max_detections=200,
    nms_mode="classwise",
    agnostic_nms_thresh=0.7,
    progress=True,
):
    model.eval()
    predictions, targets = [], []
    for images, batch_targets in tqdm(loader, desc="evaluating", disable=not progress):
        images = images.to(device)
        outputs = model(images)
        batch_preds = decode_fcos(
            outputs,
            model.strides,
            score_thresh=conf,
            nms_thresh=iou,
            max_detections=max_detections,
            nms_mode=nms_mode,
            agnostic_nms_thresh=agnostic_nms_thresh,
            regress_normalized=getattr(model, "regress_normalized", False),
        )
        for pred, target in zip(batch_preds, batch_targets):
            predictions.append({k: v.detach().cpu() for k, v in pred.items()})
            targets.append({"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()})
    metrics = evaluate_map50(predictions, targets)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics
