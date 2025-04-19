import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features.cqt import CQT2010v2
from omegaconf import DictConfig

from ..utils import get_norm, init_weights
from ..act import Swish
from torchaudio.transforms import Resample


class DiscriminatorP(nn.Module):
    def __init__(self, config: DictConfig, period: int) -> None:
        super().__init__()
        self.period = period

        self.kernel_size = config.mpd.kernel_size
        self.stride = config.mpd.stride
        self.activation = config.mpd.activation
        self.lrelu_slope = config.mpd.lrelu_slope

        norm = get_norm(config.mpd.norm)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    64,
                    (self.kernel_size, 1),
                    (self.stride, 1),
                    padding=(self.kernel_size // 2, 0),
                ),
                nn.Conv2d(
                    64,
                    128,
                    (self.kernel_size, 1),
                    (self.stride, 1),
                    padding=(self.kernel_size // 2, 0),
                ),
                nn.Conv2d(
                    128,
                    256,
                    (self.kernel_size, 1),
                    (self.stride, 1),
                    padding=(self.kernel_size // 2, 0),
                ),
                nn.Conv2d(
                    256,
                    512,
                    (self.kernel_size, 1),
                    (self.stride, 1),
                    padding=(self.kernel_size // 2, 0),
                ),
                nn.Conv2d(
                    512,
                    1024,
                    (self.kernel_size, 1),
                    1,
                    padding=(self.kernel_size // 2, 0),
                ),
            ]
        )
        self.convs.apply(init_weights)
        for i in range(len(self.convs)):
            self.convs[i] = norm(self.convs[i])

        self.activations = nn.ModuleList()
        for c in self.convs:
            match self.activation:
                case "ReLU":
                    self.activations.append(
                        nn.LeakyReLU(self.lrelu_slope, inplace=True)
                    )
                case "Swish":
                    self.activations.append(Swish(c.out_channels, dim=1, d2d=True))
                case "GELU":
                    self.activations.append(nn.GELU())
                case _:
                    raise ValueError(f"Unknown activation: {self.activation}")

        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        self.conv_post.apply(init_weights)
        self.conv_post = norm(self.conv_post)

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for c, a in zip(self.convs, self.activations):
            x = c(x)
            x = a(x)
            fmap.append(x)

        x = self.conv_post(x)
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


class DiscriminatorQ(nn.Module):
    def __init__(
        self, config: DictConfig, resolution: tuple[int, int, int], sample_rate: int
    ) -> None:
        super().__init__()
        self.hop_length, self.octaves, self.bins_per_octave = resolution

        self.filters = config.cqtd.filters
        self.dilations = config.cqtd.dilations
        self.activation = config.cqtd.activation
        self.lrelu_slope = config.cqtd.lrelu_slope

        norm = get_norm(config.cqtd.norm)

        self.cqt = CQT2010v2(
            sr=sample_rate,
            fmin=config.f_min,
            hop_length=self.hop_length,
            n_bins=self.octaves * self.bins_per_octave,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
            verbose=False,
        )

        self.band_convs = nn.ModuleList(
            [
                nn.Conv2d(2, 2, (3, 9), padding=self.get_2d_padding((3, 9)))
                for _ in range(self.octaves)
            ]
        )
        self.band_convs.apply(init_weights)
        for i in range(len(self.band_convs)):
            self.band_convs[i] = norm(self.band_convs[i])

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(2, self.filters, (3, 9), padding=self.get_2d_padding((3, 9)))
        )

        for i, d in enumerate(self.dilations):
            self.convs.append(
                nn.Conv2d(
                    self.filters,
                    self.filters,
                    (3, 9),
                    stride=(1, 2),
                    padding=self.get_2d_padding((3, 9), (d, 1)),
                    dilation=(d, 1),
                )
            )

        self.convs.append(
            nn.Conv2d(
                self.filters,
                self.filters,
                (3, 9),
                padding=self.get_2d_padding((3, 9)),
            )
        )

        self.convs.apply(init_weights)
        for i in range(len(self.convs)):
            self.convs[i] = norm(self.convs[i])

        self.activations = nn.ModuleList()
        for c in self.convs:
            match self.activation:
                case "ReLU":
                    self.activations.append(
                        nn.LeakyReLU(self.lrelu_slope, inplace=True)
                    )
                case "Swish":
                    self.activations.append(Swish(c.out_channels, dim=1, d2d=True))
                case "GELU":
                    self.activations.append(nn.GELU())
                case _:
                    raise ValueError(f"Unknown activation: {self.activation}")

        self.conv_post = nn.Conv2d(
            self.filters, 1, (3, 9), padding=self.get_2d_padding((3, 9))
        )
        self.conv_post.apply(init_weights)
        self.conv_post = norm(self.conv_post)

    def get_2d_padding(
        self,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        fmap = []
        x = self.cqt(x)
        amplitude = x[:, :, :, 0].unsqueeze(1)
        phase = x[:, :, :, 1].unsqueeze(1)
        z = torch.cat([amplitude, phase], dim=1)
        z = z.permute(0, 1, 3, 2)  # [B, C, W, T] -> [B, C, T, W]

        latent = []
        for i in range(self.octaves):
            z_band = self.band_convs[i](
                z[..., i * self.bins_per_octave : (i + 1) * self.bins_per_octave]
            )
            latent.append(z_band)

        x = torch.cat(latent, dim=-1)

        for c, a in zip(self.convs, self.activations):
            x = c(x)
            x = a(x)
            fmap.append(x)
        x = self.conv_post(x)

        return fmap, x


class MultiCQTDiscriminator(torch.nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.sample_rate = config.sample_rate
        self.resolutions = config.cqtd.resolutions
        self.resample = nn.Identity()

        if config.cqtd.resample:
            self.resample = Resample(self.sample_rate, self.sample_rate * 2)
            self.sample_rate *= 2

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorQ(config, resolution, self.sample_rate)
                for resolution in self.resolutions
            ]
        )

    def forward(self, x: torch.Tensor) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]
