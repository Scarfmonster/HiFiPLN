import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from julius import ResampleFrac
from torch.nn.utils.parametrizations import weight_norm

from alias.resample import DownSample1d, UpSample1d
from model.act import ReSnake, SnakeGamma, Swish
from model.utils import get_norm, get_padding, init_weights


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 64,
        conditiondim: int | None = None,
        dropout: float = 0.0,
        gqa: bool = False,
    ):
        super().__init__()
        if conditiondim is None:
            conditiondim = dim

        self.dropout = dropout
        self.gqa = gqa

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(conditiondim, hidden_dim * 2, bias=False)

        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # b, c, h, w = x.shape
        if kv is None:
            kv = q

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=2)

        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.heads), (q, k, v)
        )

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout,
            enable_gqa=self.gqa,
        )

        out = rearrange(
            out,
            "b h t c -> b t (h c) ",
            h=self.heads,
        )
        return self.to_out(out)


class AttenLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        transpose: bool = True,
        ff_dropout: float = 0.1,
        att_dropout: float = 0.0,
        gqa: bool = False,
    ) -> None:
        super().__init__()
        self.transpose = transpose
        self.ff_dropout = ff_dropout
        self.att_dropout = att_dropout
        self.gqa = gqa

        self.attention = Attention(
            hidden,
            heads=4,
            dim_head=32,
            dropout=self.att_dropout,
            gqa=self.gqa,
        )
        self.norm1 = nn.LayerNorm(hidden)

        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(self.ff_dropout, inplace=True),
            nn.Linear(hidden * 2, hidden),
        )
        self.ff.apply(init_weights)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            x = x.transpose(1, 2)

        a = self.attention(x)

        x = x + a
        x = self.norm1(x)

        c = self.ff(x)

        x = x + c
        x = self.norm2(x)

        if self.transpose:
            x = x.transpose(1, 2)

        return x


def grn(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    Gx = torch.norm(x, p=2, dim=2, keepdim=True)
    Nx = Gx / (Gx.mean(dim=-2, keepdim=True) + 1e-6)
    return gamma * (x * Nx) + beta + x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, features: int) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(1, features, 1))
        self.beta = nn.Parameter(torch.zeros(1, features, 1))

    def forward(self, x):
        return grn(x, self.gamma, self.beta)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
    ) -> None:
        super().__init__()

        self.lrelu_slope = lrelu_slope

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x2 = F.gelu(x)
            x2 = c1(x2)
            x2 = F.gelu(x2)
            x2 = c2(x2)
            x = x + x2
        return x


class ActivationBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        activation: str = "SnakeGamma",
        kernel_size: int = 3,
        dilation: tuple[int] = (1, 3, 5),
        snake_log: bool = False,
        upsample: bool = True,
        julius: bool = False,
        upsample_module: nn.Module | None = None,
        downsample_module: nn.Module | None = None,
        bias: bool = True,
        norm: str = "weight",
    ) -> None:
        super().__init__()
        self.dilations = len(dilation)

        norm = get_norm(norm)

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                    bias=bias,
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)
        for i in range(len(self.convs1)):
            self.convs1[i] = norm(self.convs1[i])
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                    bias=bias,
                )
                for d in dilation
            ]
        )
        self.convs2.apply(init_weights)
        for i in range(len(self.convs2)):
            self.convs2[i] = norm(self.convs2[i])

        self.activations = nn.ModuleList()
        for _ in range(len(dilation) * 2):
            match activation:
                case "SnakeGamma":
                    a = SnakeGamma(channels, logscale=snake_log)
                case "ReSnake":
                    a = ReSnake(channels, logscale=snake_log)
                case "Swish":
                    a = Swish(channels, dim=1)
                case "ReLU":
                    a = nn.LeakyReLU(0.1)
                case "GELU":
                    a = nn.GELU()
                case _:
                    raise ValueError(f"Unknown activation: {activation}")
            self.activations.append(a)

        if upsample:
            if upsample_module is not None and downsample_module is not None:
                self.upsample = upsample_module
                self.downsample = downsample_module
            elif julius:
                self.upsample = ResampleFrac(1, 2, rolloff=0.75, zeros=12)
                self.downsample = ResampleFrac(2, 1, rolloff=0.75, zeros=12)
            else:
                self.upsample = UpSample1d(channels, kernel_size=12)
                self.downsample = DownSample1d(channels, kernel_size=12)
        else:
            self.upsample = nn.Identity()
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor):
        for i in range(self.dilations):
            xn = self.upsample(x)
            xn = self.activations[2 * i](xn)
            xn = self.downsample(xn)
            xn = self.convs1[i](xn)
            xn = self.upsample(xn)
            xn = self.activations[2 * i + 1](xn)
            xn = self.downsample(xn)
            xn = self.convs2[i](xn)

            x = x + xn
        return x


class SnakeBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int] = (1, 3, 5),
        snake_log: bool = False,
        upsample: bool = True,
        norm: str = "weight",
    ):
        super().__init__()

        self.dilations = len(dilation)

        norm = get_norm(norm)

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)
        for i in range(len(self.convs1)):
            self.convs1[i] = norm(self.convs1[i])
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
                for d in dilation
            ]
        )
        self.convs2.apply(init_weights)
        for i in range(len(self.convs2)):
            self.convs2[i] = norm(self.convs2[i])

        self.snakes = nn.ModuleList()
        for _ in range(len(dilation) * 2):
            self.snakes.append(SnakeGamma(channels, logscale=snake_log))

        if upsample:
            self.upsample = UpSample1d(channels, 2)
            self.downsample = DownSample1d(channels, 2)
        else:
            self.upsample = nn.Identity()
            self.downsample = nn.Identity()

    def forward(self, x):
        for i in range(self.dilations):
            xn = self.upsample(x)
            xn = self.snakes[2 * i](xn)
            xn = self.downsample(xn)
            xn = self.convs1[i](xn)
            xn = self.snakes[2 * i + 1](xn)
            xn = self.convs2[i](xn)

            x = x + xn
        return x


class SmoothUpsample1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | None = None,
        output_padding: int = 0,
        norm: str = "none",
        bias: bool = True,
    ) -> None:
        super().__init__()
        norm = get_norm(norm)

        self.kernel = kernel_size
        self.stride = stride
        if padding is None:
            padding = (kernel_size - stride) // 2
        self.padding = padding
        self.output_padding = output_padding

        self.convt = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )

        self.convt.apply(init_weights)
        self.convt = norm(self.convt)
        if bias:
            self.bias = nn.Parameter(torch.normal(0.0, 0.01, (out_channels,)))
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "convt2_weight", torch.ones(1, 1, kernel_size) / kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.convt(x)

        ones = torch.ones(
            1,
            1,
            x.shape[-1],
            device=c.device,
            dtype=c.dtype,
            requires_grad=False,
        )
        c2 = F.conv_transpose1d(
            ones,
            self.convt2_weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

        x = (c / c2) / self.kernel

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1)

        return x


class TransposedLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)
