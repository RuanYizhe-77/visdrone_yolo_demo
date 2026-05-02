import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DetectionHead(nn.Module):
    def __init__(self, in_ch, out_ch, heatmap=False):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        if heatmap:
            nn.init.constant_(self.net[-1].bias, -2.19)

    def forward(self, x):
        return self.net(x)


class TinyCenterNet(nn.Module):
    """Compact anchor-free detector with stride-4 output."""

    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.num_classes = num_classes
        self.stride = 4
        self.backbone = nn.Sequential(
            ConvBNReLU(3, width // 2, stride=2),
            ConvBNReLU(width // 2, width, stride=2),
            ConvBNReLU(width, width),
            ConvBNReLU(width, width * 2),
            ConvBNReLU(width * 2, width * 2),
        )
        ch = width * 2
        self.heatmap = DetectionHead(ch, num_classes, heatmap=True)
        self.wh = DetectionHead(ch, 2)
        self.offset = DetectionHead(ch, 2)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "heatmap": self.heatmap(feat),
            "wh": self.wh(feat),
            "offset": self.offset(feat),
        }
