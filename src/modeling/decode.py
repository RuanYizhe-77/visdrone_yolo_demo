import torch
from torchvision.ops import batched_nms

from .assigner import locations_for_features
from .losses import flatten_outputs


def decode_fcos(outputs, strides, score_thresh=0.25, nms_thresh=0.5, topk=1000, max_detections=200):
    logits, reg, center = flatten_outputs(outputs)
    locations, _ = locations_for_features(outputs["logits"], strides, logits.device)
    results = []
    for b in range(logits.shape[0]):
        scores = torch.sqrt(torch.sigmoid(logits[b]) * torch.sigmoid(center[b])[:, None])
        scores_flat = scores.flatten()
        keep = scores_flat > score_thresh
        if keep.sum() == 0:
            results.append(_empty_result(logits.device))
            continue
        scores_keep = scores_flat[keep]
        if scores_keep.numel() > topk:
            scores_keep, order = scores_keep.topk(topk)
            idx = keep.nonzero(as_tuple=False).squeeze(1)[order]
        else:
            idx = keep.nonzero(as_tuple=False).squeeze(1)
        point_idx = torch.div(idx, logits.shape[-1], rounding_mode="floor")
        labels = idx % logits.shape[-1]
        loc = locations[point_idx]
        dist = reg[b, point_idx]
        boxes = torch.stack([loc[:, 0] - dist[:, 0], loc[:, 1] - dist[:, 1], loc[:, 0] + dist[:, 2], loc[:, 1] + dist[:, 3]], dim=1)
        boxes = boxes.clamp(min=0)
        keep_nms = batched_nms(boxes, scores_keep, labels, nms_thresh)[:max_detections]
        results.append({"boxes": boxes[keep_nms], "scores": scores_keep[keep_nms], "labels": labels[keep_nms]})
    return results


def _empty_result(device):
    return {
        "boxes": torch.zeros((0, 4), device=device),
        "scores": torch.zeros((0,), device=device),
        "labels": torch.zeros((0,), dtype=torch.long, device=device),
    }

