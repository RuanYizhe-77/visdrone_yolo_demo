import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return F.silu(x + identity, inplace=True)


class SmallResNetBackbone(nn.Module):
    """Readable ResNet-like backbone returning stride 8/16/32 features."""

    def __init__(self, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, width // 2, 3, stride=2),
            ConvBNAct(width // 2, width, 3, stride=2),
        )
        self.c3 = nn.Sequential(BasicBlock(width, width * 2, stride=2), BasicBlock(width * 2, width * 2))
        self.c4 = nn.Sequential(BasicBlock(width * 2, width * 4, stride=2), BasicBlock(width * 4, width * 4))
        self.c5 = nn.Sequential(BasicBlock(width * 4, width * 8, stride=2), BasicBlock(width * 8, width * 8))
        self.out_channels = [width * 2, width * 4, width * 8]

    def forward(self, x):
        x = self.stem(x)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return [c3, c4, c5]

