import torch
import torch.nn.functional as F

from src.common.boxes import aligned_generalized_box_iou
from .assigner import assign_fcos_targets, locations_for_features


def flatten_outputs(outputs):
    cls, reg, cen = [], [], []
    for logits, boxes, center in zip(outputs["logits"], outputs["bbox_reg"], outputs["centerness"]):
        b, c, h, w = logits.shape
        cls.append(logits.permute(0, 2, 3, 1).reshape(b, h * w, c))
        reg.append(boxes.permute(0, 2, 3, 1).reshape(b, h * w, 4))
        cen.append(center.permute(0, 2, 3, 1).reshape(b, h * w))
    return torch.cat(cls, dim=1), torch.cat(reg, dim=1), torch.cat(cen, dim=1)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.sum() if reduction == "sum" else loss.mean()


def distances_to_boxes(locations, distances):
    x, y = locations[:, 0], locations[:, 1]
    l, t, r, b = distances.unbind(-1)
    return torch.stack([x - l, y - t, x + r, y + b], dim=-1)


class FCOSLoss:
    def __init__(self, num_classes=10, center_radius=1.5, regress_normalized=False):
        self.num_classes = num_classes
        self.center_radius = float(center_radius)
        self.regress_normalized = bool(regress_normalized)

    def __call__(self, outputs, targets, strides):
        logits, pred_reg, pred_center = flatten_outputs(outputs)
        locations, stride_tensor = locations_for_features(outputs["logits"], strides, logits.device)
        labels, reg_targets, center_targets = assign_fcos_targets(
            locations,
            stride_tensor,
            targets,
            self.num_classes,
            center_radius=self.center_radius,
        )

        pos = labels != self.num_classes
        num_pos = pos.sum().clamp(min=1).float()
        cls_targets = torch.zeros_like(logits)
        if pos.any():
            pos_targets = cls_targets[pos]
            pos_targets[torch.arange(pos_targets.shape[0], device=logits.device), labels[pos]] = 1.0
            cls_targets[pos] = pos_targets
        cls_loss = sigmoid_focal_loss(logits, cls_targets) / num_pos

        if pos.any():
            repeated_locations = locations[None].expand(logits.shape[0], -1, -1)
            repeated_strides = stride_tensor[None].expand(logits.shape[0], -1)
            pred_distances = pred_reg[pos]
            if self.regress_normalized:
                pred_distances = pred_distances * repeated_strides[pos][:, None]
            pred_boxes = distances_to_boxes(repeated_locations[pos], pred_distances)
            target_boxes = distances_to_boxes(repeated_locations[pos], reg_targets[pos])
            giou = aligned_generalized_box_iou(pred_boxes, target_boxes)
            reg_loss = ((1.0 - giou) * center_targets[pos]).sum() / center_targets[pos].sum().clamp(min=1.0)
            center_loss = F.binary_cross_entropy_with_logits(pred_center[pos], center_targets[pos], reduction="sum") / num_pos
        else:
            reg_loss = pred_reg.sum() * 0.0
            center_loss = pred_center.sum() * 0.0
        total = cls_loss + reg_loss + center_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "center_loss": center_loss.detach(),
            "num_pos": num_pos.detach(),
        }
