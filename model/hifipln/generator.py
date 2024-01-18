import math

import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from alias.act import Activation1d
from alias.resample import DownSample1d
from model.common import SnakeBlock, SnakeGamma

from ..ddsp.generator import DDSP
from ..utils import init_weights
from .encoder import PreEncoder


class HiFiPLN(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.n_mels = config.n_mels
        self.upsample_initial = config.model.upsample_initial

        self.source = DDSP(config, layers=1)
        self.encoder = PreEncoder(config, self.n_mels + self.upsample_initial)
        self.updown_block = UpDownSampleBlock(config)

    def forward(self, x, f0):
        x = self.encoder(x, f0)
        src = x[:, self.upsample_initial :]
        x = x[:, : self.upsample_initial]
        src, (src_harmonic, src_noise) = self.source(src, f0)
        waveform = self.updown_block(x, src)

        return waveform, (src_harmonic, src_noise)

    def remove_parametrizations(self):
        param = 0
        for module in self.modules():
            if hasattr(module, "weight") and is_parametrized(module, "weight"):
                param += 1
                remove_parametrizations(module, "weight")
        print(f"Removed {param} parametrizations.")


class UpDownSampleBlock(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.n_mels = config.n_mels
        self.snake_log = config.model.snake_log
        self.snake_upsample = config.model.snake_upsample
        self.upsample_initial = config.model.upsample_initial
        self.upsample_rates = config.model.upsample_rates
        self.upsample_num = len(self.upsample_rates)
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

        for i in range(self.upsample_num):
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
                        upsample=self.snake_upsample,
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
                    )
                )
            )
            self.snakes.append(snake(in_ch, logscale=self.snake_log))

        self.post_snake = Activation1d(snake(out_ch, logscale=self.snake_log), out_ch)
        self.post_conv = weight_norm(nn.Conv1d(out_ch, 1, 7, padding=3))

        self.pre_conv.apply(init_weights)
        self.upsamples.apply(init_weights)
        self.source_conv.apply(init_weights)
        self.post_conv.apply(init_weights)

    def forward(self, x, source):
        x = self.pre_conv(x)

        for i in range(self.upsample_num):
            x = self.snakes[i](x)
            x = self.upsamples[i](x)

            source_x = self.downsamples[i](source)
            source_x = self.source_conv[i](source_x)
            # x = x + source_x

            xn = None
            for j in range(self.kernels):
                x2 = self.convs[self.kernels * i + j](x)
                if xn is None:
                    xn = x2
                else:
                    xn += x2

                xn = xn + source_x

            x = xn / self.kernels

        x = self.post_snake(x)
        x = self.post_conv(x)
        x = F.hardtanh(x, -1, 1)

        return x
