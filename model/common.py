import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from alias.act import Activation1d

from model.utils import get_padding, init_weights


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
    ):
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
            x2 = F.leaky_relu(x, self.lrelu_slope, inplace=False)
            x2 = c1(x2)
            x2 = F.leaky_relu(x2, self.lrelu_slope, inplace=True)
            x2 = c2(x2)
            x = x + x2
        return x


class SnakeBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int] = (1, 3, 5),
        snake_log: bool = False,
    ):
        super().__init__()

        self.dilations = len(dilation)

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

        self.snakes = nn.ModuleList()
        for _ in range(len(dilation) * 2):
            self.snakes.append(
                Activation1d(SnakeGamma(channels, logscale=snake_log), channels)
            )

    def forward(self, x):
        xn = None
        for i in range(self.dilations):
            x2 = self.snakes[2 * i](x)
            x2 = self.convs1[i](x2)
            x2 = self.snakes[2 * i + 1](x2)
            x2 = self.convs2[i](x2)
            if xn is None:
                xn = x2
            else:
                xn += x2
        x = x + xn / self.dilations
        return x


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, logscale=False):
        super().__init__()
        self.in_features = in_features

        if logscale:
            beta = torch.log(torch.normal(alpha, 0.1, size=(in_features,)))
            alpha = torch.log(torch.normal(alpha, 0.2, size=(in_features,)))

            self.alpha = nn.Parameter(torch.zeros(in_features) + alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) + beta)
            self.exp = Exp()
        else:
            self.alpha = nn.Parameter(torch.normal(alpha, 0.2, size=(in_features,)))
            self.beta = nn.Parameter(torch.normal(alpha, 0.1, size=(in_features,)))
            self.exp = nn.Identity()

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)

        alpha = self.exp(alpha)
        beta = self.exp(beta)

        x = x + (1.0 / (beta + 1e-8)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class SnakeGamma(nn.Module):
    def __init__(self, in_features, alpha=1.0, logscale=False):
        super().__init__()
        self.in_features = in_features

        if logscale:
            beta = torch.log(torch.normal(alpha, 0.1, size=(in_features,)))
            alpha = torch.log(torch.normal(alpha, 0.2, size=(in_features,)))

            self.alpha = nn.Parameter(torch.zeros(in_features) + alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) + beta)
            self.exp = Exp()
        else:
            self.alpha = nn.Parameter(torch.normal(alpha, 0.2, size=(in_features,)))
            self.beta = nn.Parameter(torch.normal(alpha, 0.1, size=(in_features,)))
            self.exp = nn.Identity()

        self.gamma = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        gamma = self.gamma.unsqueeze(0).unsqueeze(-1)

        alpha = self.exp(alpha)
        beta = self.exp(beta)

        x = x * gamma + (1.0 / (beta + 1e-8)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)
