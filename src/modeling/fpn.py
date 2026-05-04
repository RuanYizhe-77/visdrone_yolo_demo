from torch import nn
import torch.nn.functional as F

from .backbone import ConvBNAct


class SimpleFPN(nn.Module):
    def __init__(self, in_channels, out_ch=160):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.output = nn.ModuleList([ConvBNAct(out_ch, out_ch) for _ in in_channels])

    def forward(self, feats):
        pyramid = [None for _ in feats]
        last = self.lateral[-1](feats[-1])
        pyramid[-1] = last
        for idx in range(len(feats) - 2, -1, -1):
            last = self.lateral[idx](feats[idx]) + F.interpolate(last, size=feats[idx].shape[-2:], mode="nearest")
            pyramid[idx] = last
        return [out(feat) for out, feat in zip(self.output, pyramid)]
