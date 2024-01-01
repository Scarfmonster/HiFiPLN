import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from alias.act import Activation1d
from alias.resample import DownSample1d
from model.common import SnakeGamma, SnakeBlock
from model.ddsp.generator import DDSP

from ..utils import init_weights
from .source import DDSP


class HiFiPLN(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.pre_encoder = PreEncoder(config)
        self.source = DDSP(config)

        self.harmonic_block = HarmonicBlock(config)
        self.noise_block = NoiseBlock(config)

    def forward(self, x, f0):
        _, (harmonic, noise) = self.source(x, f0)

        x = self.pre_encoder(x)
        harmonic = self.harmonic_block(x, harmonic)
        noise = self.noise_block(x, noise)

        waveform = harmonic + noise
        waveform = F.hardtanh(waveform, -1, 1)

        return waveform, (harmonic, noise)

    def remove_parametrizations(self):
        param = 0
        for module in self.modules():
            if hasattr(module, "weight") and is_parametrized(module, "weight"):
                param += 1
                remove_parametrizations(module, "weight")
        print(f"Removed {param} parametrizations.")

    def prune(self):
        self.harmonic_block.prune()
        self.noise_block.prune()


class PreEncoder(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.n_mels = config.n_mels
        self.upsample_initial = config.model.upsample_initial
        self.snake_log = config.model.snake_log

        steps = self.upsample_initial // self.n_mels + 1

        self.convs = nn.ModuleList()
        self.glus = nn.ModuleList()

        for i in range(steps):
            in_ch = self.n_mels * (2**i)
            out_ch = self.n_mels * (2 ** (i + 1)) * 2
            self.convs.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    3,
                    padding=1,
                )
            )
            self.glus.append(nn.GLU(1))

        self.post_norm = nn.LayerNorm(self.n_mels * (2**steps))

        self.post = nn.Linear(
            self.n_mels * (2**steps), self.upsample_initial, bias=False
        )

        self.convs.apply(init_weights)

    def forward(self, x):
        for conv, glu in zip(self.convs, self.glus):
            x = conv(x)
            x = glu(x)

        x = x.transpose(1, 2)
        x = self.post_norm(x)
        x = self.post(x)
        x = x.transpose(1, 2)

        return x


class HarmonicBlock(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.n_mels = config.n_mels
        self.snake_log = config.model.snake_log
        self.upsample_initial = config.model.upsample_initial
        self.upsample_rates = config.model.upsample_rates
        self.upsample_kernels = config.model.upsample_kernels
        self.kernel_sizes = config.model.kernel_sizes
        self.dilation_sizes = config.model.dilation_sizes
        self.kernels = len(self.kernel_sizes)

        assert len(self.upsample_rates) == len(self.upsample_kernels)

        self.pre_conv = weight_norm(
            nn.Conv1d(self.upsample_initial, self.upsample_initial, 7, padding=3)
        )

        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.source_conv = nn.ModuleList()
        self.snakes = nn.ModuleList()

        snake = SnakeGamma

        for i in range(len(self.upsample_rates)):
            in_ch = self.upsample_initial // (2**i)
            out_ch = self.upsample_initial // (2 ** (i + 1))
            kernel = self.upsample_kernels[i]
            rate = self.upsample_rates[i]

            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )
            for k, d in zip(self.kernel_sizes, self.dilation_sizes):
                self.convs.append(
                    SnakeBlock(
                        out_ch,
                        kernel_size=k,
                        dilation=d,
                        snake_log=self.snake_log,
                    )
                )

            downs_rate = math.prod(self.upsample_rates[i + 1 :])
            if downs_rate > 1:
                self.downsamples.append(DownSample1d(1, downs_rate))
            else:
                self.downsamples.append(nn.Identity())
            self.source_conv.append(
                weight_norm(
                    nn.Conv1d(
                        1,
                        out_ch,
                        kernel + 1,
                        padding=(kernel + 1) // 2,
                        bias=False,
                    )
                )
            )
            self.snakes.append(
                Activation1d(snake(in_ch, logscale=self.snake_log), in_ch)
            )

        self.post_snake = Activation1d(snake(out_ch, logscale=self.snake_log), out_ch)
        self.post_conv = weight_norm(nn.Conv1d(out_ch, 1, 7, padding=3, bias=False))

        self.pre_conv.apply(init_weights)
        self.upsamples.apply(init_weights)
        self.convs.apply(init_weights)
        self.source_conv.apply(init_weights)
        self.post_conv.apply(init_weights)

    def forward(self, x, source):
        x = self.pre_conv(x)

        for i in range(len(self.upsample_rates)):
            x = self.snakes[i](x)
            x = self.upsamples[i](x)

            source_x = self.downsamples[i](source)
            source_x = self.source_conv[i](source_x)

            x = x + source_x

            xn = None
            for j in range(self.kernels):
                x2 = self.convs[self.kernels * i + j](x)
                if xn is None:
                    xn = x2
                else:
                    xn += x2

            x = x + xn / self.kernels

        x = self.post_snake(x)
        x = self.post_conv(x)
        x = F.tanh(x)

        return x


class NoiseBlock(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.n_mels = config.n_mels
        self.snake_log = config.model.snake_log
        self.upsample_initial = config.model.upsample_initial
        self.upsample_rates = config.model.upsample_rates
        self.upsample_kernels = config.model.upsample_kernels
        self.res_kernel = config.model.kernel_sizes[-1]
        self.res_dilation = config.model.dilation_sizes[-1]

        snake = SnakeGamma

        self.pre_conv = weight_norm(
            nn.Conv1d(self.upsample_initial, self.upsample_initial, 7, padding=3)
        )

        self.upsamples = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.snakes = nn.ModuleList()

        for i in range(len(self.upsample_rates)):
            in_ch = self.upsample_initial // (2**i)
            out_ch = self.upsample_initial // (2 ** (i + 1))
            rate = self.upsample_rates[i]
            kernel = self.upsample_kernels[i]

            self.snakes.append(
                Activation1d(snake(in_ch, logscale=self.snake_log), in_ch)
            )

            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )

            downs_rate = math.prod(self.upsample_rates[i + 1 :])
            if downs_rate > 1:
                self.downsamples.append(
                    nn.Conv1d(
                        1,
                        out_ch,
                        kernel_size=downs_rate * 2,
                        stride=downs_rate,
                        padding=downs_rate // 2,
                    )
                )
            else:
                self.downsamples.append(nn.Conv1d(1, out_ch, 7, padding=3))

        self.conv = SnakeBlock(
            out_ch,
            kernel_size=self.res_kernel,
            dilation=self.res_dilation,
            snake_log=self.snake_log,
        )

        self.post_conv = weight_norm(nn.Conv1d(out_ch, 1, 7, padding=3, bias=False))

    def forward(self, x, source):
        x = self.pre_conv(x)

        for i in range(len(self.upsample_rates)):
            x = self.snakes[i](x)
            x = self.upsamples[i](x)

            source_x = self.downsamples[i](source)
            x = x + source_x

        x = self.conv(x)

        x = self.post_conv(x)
        x = F.tanh(x)

        return x
