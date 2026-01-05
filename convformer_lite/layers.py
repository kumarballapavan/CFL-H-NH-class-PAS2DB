import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """Conv + BN + SiLU stem, with stride-2 downsampling."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvMixerBlock(nn.Module):
    """
    ConvMixer-style block:
      DWConv(k=7) -> BN -> GELU -> PWConv(1x1) -> BN -> GELU + residual
    """
    def __init__(self, dim: int, k: int = 7, drop: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=k, padding=k//2, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.act(self.bn1(self.dw(x)))
        y = self.drop(y)
        y = self.act(self.bn2(self.pw(y)))
        return x + y


class SafeMHSA2D(nn.Module):
    """
    Multi-head self-attention over 2D features with a token budget.
    If H*W > token_budget, it pools to approx sqrt(token_budget) x sqrt(token_budget),
    applies attention, then upsamples back to original size.

    This keeps attention compute bounded for lightweight models.
    """
    def __init__(self, dim: int, num_heads: int = 4, token_budget: int = 256, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.token_budget = int(token_budget)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm = nn.LayerNorm(dim)

    def _pool_if_needed(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        n = H * W
        if n <= self.token_budget:
            return x, (H, W), False

        side = int(math.sqrt(self.token_budget))
        side = max(1, side)
        pooled = F.adaptive_avg_pool2d(x, output_size=(side, side))
        return pooled, (H, W), True

    def forward(self, x):
        # x: [B, C, H, W]
        x_in = x
        x, (H0, W0), pooled = self._pool_if_needed(x)

        B, C, H, W = x.shape
        n = H * W

        # tokens: [B, N, C]
        t = x.flatten(2).transpose(1, 2)
        t = self.norm(t)

        qkv = self.qkv(t)  # [B, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, heads, N, head_dim]
        q = q.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, n, C)
        out = self.proj_drop(self.proj(out))

        # back to map: [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)

        if pooled:
            out = F.interpolate(out, size=(H0, W0), mode="bilinear", align_corners=False)

        return x_in + out
