import torch
from torch import nn
import torch.nn.functional as F


class CenterNetLoss(nn.Module):
    def __init__(self, wh_weight=0.1, off_weight=1.0):
        super().__init__()
        self.wh_weight = wh_weight
        self.off_weight = off_weight

    def forward(self, pred, target):
        # BCEWithLogits is stable and adequate for this interview demo.
        hm_loss = F.binary_cross_entropy_with_logits(pred["heatmap"], target["heatmap"])
        mask = target["mask"].unsqueeze(1)
        denom = mask.sum().clamp(min=1.0)
        wh_loss = F.l1_loss(pred["wh"] * mask, target["wh"] * mask, reduction="sum") / denom
        off_loss = F.l1_loss(pred["offset"] * mask, target["offset"] * mask, reduction="sum") / denom
        total = hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss
        return {
            "loss": total,
            "heatmap_loss": hm_loss.detach(),
            "wh_loss": wh_loss.detach(),
            "offset_loss": off_loss.detach(),
        }
