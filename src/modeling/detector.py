import math

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import ConvBNAct, SmallResNetBackbone
from .fpn import SimpleFPN


class FCOSHead(nn.Module):
    def __init__(self, in_ch=160, num_classes=10, num_convs=3, reg_activation="relu"):
        super().__init__()
        self.reg_activation = reg_activation
        cls_layers, box_layers = [], []
        for _ in range(num_convs):
            cls_layers.append(ConvBNAct(in_ch, in_ch))
            box_layers.append(ConvBNAct(in_ch, in_ch))
        self.cls_tower = nn.Sequential(*cls_layers)
        self.box_tower = nn.Sequential(*box_layers)
        self.cls_logits = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.bbox_reg = nn.Conv2d(in_ch, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_ch, 1, 3, padding=1)
        prior_prob = 0.01
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, features):
        logits, boxes, centers = [], [], []
        for feat in features:
            cls_feat = self.cls_tower(feat)
            box_feat = self.box_tower(feat)
            logits.append(self.cls_logits(cls_feat))
            raw_boxes = self.bbox_reg(box_feat)
            if self.reg_activation == "softplus":
                boxes.append(F.softplus(raw_boxes))
            else:
                boxes.append(torch.relu(raw_boxes))
            centers.append(self.centerness(box_feat))
        return {"logits": logits, "bbox_reg": boxes, "centerness": centers}


class FCOSLiteDetector(nn.Module):
    def __init__(
        self,
        num_classes=10,
        width=64,
        fpn_channels=160,
        head_convs=3,
        use_p2=False,
        reg_activation="relu",
        regress_normalized=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_p2 = bool(use_p2)
        self.regress_normalized = bool(regress_normalized)
        self.reg_activation = reg_activation
        self.strides = [4, 8, 16, 32] if self.use_p2 else [8, 16, 32]
        self.backbone = SmallResNetBackbone(width=width, use_p2=self.use_p2)
        self.fpn = SimpleFPN(self.backbone.out_channels, out_ch=fpn_channels)
        self.head = FCOSHead(fpn_channels, num_classes=num_classes, num_convs=head_convs, reg_activation=reg_activation)

    def forward(self, x):
        return self.head(self.fpn(self.backbone(x)))
