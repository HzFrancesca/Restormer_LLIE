import torch
import torch.nn as nn
from einops import rearrange


# WxW
class HTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(HTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head (c h) w", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head (c h) w", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head (c h) w", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)

        attn = (q.transpose(-2, -1) @ k) * self.temperature  # wxw
        attn = attn.softmax(dim=-2)

        out = v @ attn

        out = rearrange(
            out, "b head (c h) w -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


# HxH
class WTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(WTA, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        v1 = rearrange(v, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        q1 = rearrange(q, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        k1 = rearrange(k, "b (head c) h w -> b head h (c w)", head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1) * self.temperature).softmax(dim=-1)

        out = attn1 @ v1
        out = rearrange(
            out, "b head h (c w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out


# CxHxH - Intra-channel row attention
class IRS(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(IRS, self).__init__()
        self.temperature = nn.Parameter(torch.ones(dim, 1, 1))
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q1 = torch.nn.functional.normalize(q, dim=-1)
        k1 = torch.nn.functional.normalize(k, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        out = attn1 @ v
        out = self.project_out(out)
        return out


# CxWxW - Intra-channel column attention
class ICS(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(ICS, self).__init__()
        self.temperature = nn.Parameter(torch.ones(dim, 1, 1))
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q1 = torch.nn.functional.normalize(q, dim=-2)
        k1 = torch.nn.functional.normalize(k, dim=-2)
        attn1 = (q1.transpose(-2, -1) @ k1 * self.temperature).softmax(dim=-2)  # CWW
        out = v @ attn1  # CHW@CWW -> CHW
        out = self.project_out(out)
        return out
