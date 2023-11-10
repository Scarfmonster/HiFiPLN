import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils.parametrizations import weight_norm

from ..utils import get_padding, init_weights


class DiscriminatorP(nn.Module):
    def __init__(self, config: DictConfig, period: int) -> None:
        super().__init__()
        self.period = period

        self.lrelu_slope = config.mpd.lrelu_slope
        self.kernel_size = config.mpd.kernel_size
        self.stride = config.mpd.stride

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        1,
                        64,
                        (self.kernel_size, 1),
                        (self.stride, 1),
                        padding=(get_padding(self.kernel_size), 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        64,
                        128,
                        (self.kernel_size, 1),
                        (self.stride, 1),
                        padding=(get_padding(self.kernel_size), 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        128,
                        256,
                        (self.kernel_size, 1),
                        (self.stride, 1),
                        padding=(get_padding(self.kernel_size), 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        256,
                        512,
                        (self.kernel_size, 1),
                        (self.stride, 1),
                        padding=(get_padding(self.kernel_size), 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        512,
                        1024,
                        (self.kernel_size, 1),
                        1,
                        padding=(get_padding(self.kernel_size), 0),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.conv_post.apply(init_weights)

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
            x = torch.nan_to_num(x)

            fmap.append(x)

        x = self.conv_post(x)
        x = torch.nan_to_num(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.periods = config.mpd.periods
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(config, period))

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret


class DiscriminatorR(nn.Module):
    def __init__(self, config: DictConfig, resolution):
        super().__init__()

        self.lrelu_slope = config.mrd.lrelu_slope
        self.resolution = resolution

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.convs.apply(init_weights)

        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))
        self.conv_post.apply(init_weights)

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True,
        )  # [B, F, TT, 2]
        x = torch.view_as_real(x)
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resolutions = config.mrd.resolutions
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(config, resolution) for resolution in self.resolutions]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]
