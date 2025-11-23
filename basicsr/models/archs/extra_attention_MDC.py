import torch
import torch.nn as nn
from einops import rearrange


class MDC(nn.Module):
    def __init__(self, dim, bias):
        super(MDC, self).__init__()

        self.mdc1 = nn.Conv2d(
            dim, int(dim / 2), groups=int(dim / 2), kernel_size=3, bias=bias, padding=1
        )
        self.mdc2 = nn.Conv2d(
            int(dim / 2),
            int(dim / 2),
            groups=int(dim / 2),
            kernel_size=3,
            bias=bias,
            padding=1,
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.mdc1(x)
        x2 = self.mdc2(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.sig(x) * out
        return out


class HTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(HTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.mdc_q = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.mdc_k = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.mdc_v = MDC(dim, bias)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.mdc_q(q), self.mdc_k(k), self.mdc_v(v)
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


class WTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(WTA, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.mdc_q = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.mdc_k = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.mdc_v = MDC(dim, bias)
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        _, _, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.mdc_q(q), self.mdc_k(k), self.mdc_v(v)
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


class IRS(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(IRS, self).__init__()
        self.temperature = nn.Parameter(torch.ones(dim, 1, 1))

        self.mdc_q = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.mdc_k = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.mdc_v = MDC(dim, bias)
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.mdc_q(q), self.mdc_k(k), self.mdc_v(v)
        _, _, h, w = x.shape
        q1 = torch.nn.functional.normalize(q, dim=-1)
        k1 = torch.nn.functional.normalize(k, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        out = attn1 @ v
        out = self.project_out(out)
        return out


class ICS(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(ICS, self).__init__()
        self.temperature = nn.Parameter(torch.ones(dim, 1, 1))
        self.mdc_q = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.mdc_k = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.mdc_v = MDC(dim, bias)
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.mdc_q(q), self.mdc_k(k), self.mdc_v(v)  # CHW
        _, _, h, w = x.shape
        q1 = torch.nn.functional.normalize(q, dim=-2)
        k1 = torch.nn.functional.normalize(k, dim=-2)
        attn1 = (q1.transpose(-2, -1) @ k1 * self.temperature).softmax(dim=-2)  # CWW
        out = v @ attn1  # CHW@CWW -> CHW
        out = self.project_out(out)
        return out
