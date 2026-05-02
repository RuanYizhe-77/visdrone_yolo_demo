import torch
from torchvision.ops import nms


def decode_predictions(pred, conf_thresh=0.3, iou_thresh=0.5, stride=4, topk=200):
    heatmap = pred["heatmap"].sigmoid()
    wh = pred["wh"].clamp(min=0)
    offset = pred["offset"]
    b, c, h, w = heatmap.shape
    results = []

    for bi in range(b):
        scores, inds = torch.topk(heatmap[bi].reshape(-1), k=min(topk, c * h * w))
        keep = scores >= conf_thresh
        scores, inds = scores[keep], inds[keep]
        if scores.numel() == 0:
            results.append({"boxes": torch.empty((0, 4), device=heatmap.device), "scores": scores, "labels": inds})
            continue

        labels = inds // (h * w)
        rem = inds % (h * w)
        ys = rem // w
        xs = rem % w
        wh_vals = wh[bi, :, ys, xs].permute(1, 0)
        off_vals = offset[bi, :, ys, xs].permute(1, 0)
        centers = (torch.stack([xs, ys], dim=1).float() + off_vals) * stride
        sizes = wh_vals * stride
        boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=1)

        final = []
        for cls in labels.unique():
            cls_idx = torch.where(labels == cls)[0]
            cls_keep = nms(boxes[cls_idx], scores[cls_idx], iou_thresh)
            final.append(cls_idx[cls_keep])
        final = torch.cat(final) if final else torch.empty(0, dtype=torch.long, device=heatmap.device)
        results.append({"boxes": boxes[final], "scores": scores[final], "labels": labels[final]})
    return results
