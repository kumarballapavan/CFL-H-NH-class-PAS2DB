import torch.nn as nn
from .layers import ConvStem, ConvMixerBlock, SafeMHSA2D


class ConvFormerLite(nn.Module):
    """
    Lightweight ConvFormer-Lite:
      - Conv stem
      - Stage-wise ConvMixer blocks
      - Periodic SafeMHSA2D blocks (token budget)
      - Global average pooling + linear head
    """
    def __init__(
        self,
        num_classes: int = 2,
        widths=(48, 96, 192),
        depths=(2, 3, 2),
        attn_every: int = 2,
        token_budget: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
        in_ch: int = 3,
    ):
        super().__init__()
        assert len(widths) == len(depths), "widths and depths must match"

        self.stem = ConvStem(in_ch, widths[0])

        stages = []
        for si, (dim, depth) in enumerate(zip(widths, depths)):
            # downsample at stage start (except stage0, already downsampled by stem)
            if si > 0:
                stages.append(nn.Conv2d(widths[si - 1], dim, kernel_size=3, stride=2, padding=1, bias=False))
                stages.append(nn.BatchNorm2d(dim))
                stages.append(nn.SiLU(inplace=True))

            # blocks
            for bi in range(depth):
                stages.append(ConvMixerBlock(dim, k=7, drop=dropout))
                if attn_every > 0 and ((bi + 1) % attn_every == 0):
                    stages.append(SafeMHSA2D(dim, num_heads=num_heads, token_budget=token_budget, dropout=dropout))

        self.backbone = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
