from torch import nn
import torch.nn.functional as F

from .backbone import ConvBNAct


class SimpleFPN(nn.Module):
    def __init__(self, in_channels, out_ch=160):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.output = nn.ModuleList([ConvBNAct(out_ch, out_ch) for _ in in_channels])

    def forward(self, feats):
        c3, c4, c5 = feats
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return [self.output[0](p3), self.output[1](p4), self.output[2](p5)]

