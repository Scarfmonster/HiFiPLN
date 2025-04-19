# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0

import torch
import torch.nn as nn
from torch.nn import functional as F
from .filter import LowPassFilter1d
from .filter import kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    def __init__(self, channels: int, ratio=2, kernel_size=None):
        super().__init__()
        self.channels = channels
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        x = upsample1d(
            x,
            self.filter,
            self.ratio,
            self.channels,
            self.stride,
            self.pad,
            self.pad_left,
            self.pad_right,
        )

        return x


@torch.jit.script_if_tracing
def upsample1d(
    x,
    filter: torch.Tensor,
    ratio: int,
    channels: int,
    stride: int,
    pad: int,
    pad_left: int,
    pad_right: int,
):
    x = F.pad(x, (pad, pad), mode="replicate")
    x = ratio * F.conv_transpose1d(
        x, filter.expand(channels, -1, -1), stride=stride, groups=channels
    )
    x = x[..., pad_left:-pad_right]

    return x


class DownSample1d(nn.Module):
    def __init__(self, channels: int, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            channels,
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        xx = self.lowpass(x)

        return xx
