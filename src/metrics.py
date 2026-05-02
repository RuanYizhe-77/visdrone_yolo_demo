import numpy as np
from torchvision.ops import box_iou


def evaluate_detections(all_preds, all_targets, num_classes=10, iou_thresh=0.5):
    per_class = []
    aps = []
    total_tp = total_fp = total_fn = 0

    for cls in range(num_classes):
        cls_preds = []
        cls_gts = {}
        for img_id, (pred, target) in enumerate(zip(all_preds, all_targets)):
            p_mask = pred["labels"] == cls
            for box, score in zip(pred["boxes"][p_mask], pred["scores"][p_mask]):
                cls_preds.append((img_id, float(score), box))
            gt_boxes = target["boxes"][target["labels"] == cls]
            cls_gts[img_id] = {"boxes": gt_boxes, "matched": np.zeros(len(gt_boxes), dtype=bool)}

        cls_preds.sort(key=lambda x: x[1], reverse=True)
        tp, fp = [], []
        for img_id, _, box in cls_preds:
            g = cls_gts[img_id]
            if len(g["boxes"]) == 0:
                tp.append(0)
                fp.append(1)
                continue
            ious = box_iou(box[None], g["boxes"]).cpu().numpy()[0]
            best = int(ious.argmax())
            if ious[best] >= iou_thresh and not g["matched"][best]:
                g["matched"][best] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        tp = np.array(tp, dtype=np.float32)
        fp = np.array(fp, dtype=np.float32)
        n_gt = sum(len(v["boxes"]) for v in cls_gts.values())
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall_curve = cum_tp / max(n_gt, 1)
        precision_curve = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap = average_precision(recall_curve, precision_curve) if n_gt > 0 else 0.0
        cls_tp = int(cum_tp[-1]) if len(cum_tp) else 0
        cls_fp = int(cum_fp[-1]) if len(cum_fp) else 0
        cls_fn = n_gt - cls_tp
        total_tp += cls_tp
        total_fp += cls_fp
        total_fn += cls_fn
        aps.append(ap)
        per_class.append({"class_id": cls, "ap50": float(ap), "gt": int(n_gt), "tp": cls_tp, "fp": cls_fp, "fn": int(cls_fn)})

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "mAP50_approx": float(np.mean(aps)),
        "per_class": per_class,
        "note": "AP50 is a simplified confidence-sorted approximation, not COCO mAP.",
    }


def average_precision(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
