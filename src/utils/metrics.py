import numpy as np
import torch
from torchvision.ops import box_iou


def average_precision(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map50(predictions, targets, num_classes=10, iou_thresh=0.5):
    total_tp = total_fp = total_fn = 0
    per_class = []
    aps = []
    for cls in range(num_classes):
        preds = []
        gt_by_image = {}
        for image_id, (pred, target) in enumerate(zip(predictions, targets)):
            labels = pred["labels"].detach().cpu()
            boxes = pred["boxes"].detach().cpu()
            scores = pred["scores"].detach().cpu()
            for box, score in zip(boxes[labels == cls], scores[labels == cls]):
                preds.append((image_id, float(score), box))
            gt_boxes = target["boxes"].detach().cpu()[target["labels"].detach().cpu() == cls]
            gt_by_image[image_id] = {"boxes": gt_boxes, "matched": np.zeros(len(gt_boxes), dtype=bool)}

        preds.sort(key=lambda item: item[1], reverse=True)
        tp, fp = [], []
        for image_id, _, box in preds:
            gt = gt_by_image[image_id]
            if len(gt["boxes"]) == 0:
                tp.append(0)
                fp.append(1)
                continue
            ious = box_iou(box[None], gt["boxes"])[0].numpy()
            best = int(ious.argmax())
            if ious[best] >= iou_thresh and not gt["matched"][best]:
                gt["matched"][best] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        n_gt = sum(len(v["boxes"]) for v in gt_by_image.values())
        tp = np.asarray(tp, dtype=np.float32)
        fp = np.asarray(fp, dtype=np.float32)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(n_gt, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap = average_precision(recalls, precisions) if n_gt else 0.0
        cls_tp = int(cum_tp[-1]) if len(cum_tp) else 0
        cls_fp = int(cum_fp[-1]) if len(cum_fp) else 0
        cls_fn = int(n_gt - cls_tp)
        total_tp += cls_tp
        total_fp += cls_fp
        total_fn += cls_fn
        aps.append(ap)
        per_class.append({"class_id": cls, "ap50": ap, "gt": int(n_gt), "tp": cls_tp, "fp": cls_fp, "fn": cls_fn})

    return {
        "precision": total_tp / max(total_tp + total_fp, 1),
        "recall": total_tp / max(total_tp + total_fn, 1),
        "mAP50_approx": float(np.mean(aps)),
        "per_class": per_class,
        "note": "Simplified AP50 evaluator for learning/demo use, not official VisDrone or COCO mAP.",
    }

