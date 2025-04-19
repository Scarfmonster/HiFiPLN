import math
from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from alias.act import Activation1d

from ..act import ReSnake, SnakeGamma, Swish
from ..common import f0_to_phase, normalize
from ..layers import ActivationBlock, SmoothUpsample1D
from ..stft import STFT
from ..utils import get_norm, init_weights
from .encoder import Decoder, Encoder


class HiFiPLNv2(nn.Module):
    def __init__(self, config: DictConfig, export: bool = False) -> None:
        super().__init__()
        self.export = export
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.window = torch.hann_window(self.win_length)
        self.normalize = config.norm.get("normalize", False)
        self.upsample_rates = config.model.upsample_rates
        self.upsample_kernels = config.model.upsample_kernels
        self.upsample_initial = config.model.upsample_initial
        self.activation = config.model.activation
        self.act_log = config.model.act_log
        self.act_upsample = config.model.act_upsample
        self.decoder_hidden = config.model.decoder_hidden
        self.decoder_activation = config.model.decoder_activation
        self.detach_stft = config.model.detach_stft

        if self.normalize:
            self.register_buffer(
                "mel_mean", torch.tensor(config.norm.mel_mean).unsqueeze(0).unsqueeze(2)
            )
            self.register_buffer(
                "mel_std", torch.tensor(config.norm.mel_std).unsqueeze(0).unsqueeze(2)
            )
            self.register_buffer(
                "pitch_mean",
                torch.tensor(config.norm.pitch_mean),
            )
            self.register_buffer(
                "pitch_std",
                torch.tensor(config.norm.pitch_std),
            )

            assert (
                self.mel_mean.shape[1] == config.n_mels
            ), f"{self.mel_mean.shape[1]} != {config.n_mels}"
            assert (
                self.mel_std.shape[1] == config.n_mels
            ), f"{self.mel_std.shape[1]} != {config.n_mels}"

        norm = get_norm(config.model.norm)

        self.stft = STFT(
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            center=True,
        )

        self.encoder = Encoder(config)

        self.decoder_harmonic = Decoder(config)
        self.decoder_noise = Decoder(config)

        dim = self.n_fft // 2 + 1

        self.harmonic_out = nn.Conv1d(self.decoder_hidden, dim * 2, 1)
        self.harmonic_out.apply(init_weights)
        self.harmonic_out = norm(self.harmonic_out)

        self.noise_out = nn.Conv1d(self.decoder_hidden, dim, 1)
        self.noise_out.apply(init_weights)
        self.noise_out = norm(self.noise_out)

        self.merge = nn.Conv1d(self.decoder_hidden * 2, self.upsample_initial, 1)
        self.merge.apply(init_weights)
        self.merge = norm(self.merge)

        self.layers = nn.ModuleList()

        for i in range(len(self.upsample_rates)):
            in_channels = self.upsample_initial // (2**i)
            out_channels = self.upsample_initial // (2 ** (i + 1))
            self.layers.append(
                UpsampleLayer(
                    config,
                    in_channels=in_channels,
                    kernel=self.upsample_kernels[i],
                    rate=self.upsample_rates[i],
                    source_rate=math.prod(self.upsample_rates[i + 1 :]),
                    conv_kernels=config.model.kernel_sizes,
                    conv_dilations=config.model.dilation_sizes,
                )
            )

        match self.activation:
            case "SnakeGamma":
                self.post_act = SnakeGamma(out_channels, logscale=self.act_log)
                self.post_act = Activation1d(self.post_act, out_channels)
            case "ReSnake":
                self.post_act = ReSnake(out_channels, logscale=self.act_log)
                self.post_act = Activation1d(self.post_act, out_channels)
            case "Swish":
                self.post_act = Swish(out_channels, dim=1)
            case "ReLU":
                self.post_act = nn.LeakyReLU(0.1)
            case "GELU":
                self.post_act = nn.GELU()
            case _:
                raise ValueError(f"Unknown activation: {self.activation}")

        self.out = nn.Conv1d(out_channels, 1, 7, padding=3)
        self.out.apply(init_weights)
        self.out = norm(self.out)

    def forward(self, x: torch.Tensor, f0_frames: torch.Tensor) -> torch.Tensor:
        f0 = F.interpolate(
            torch.cat((f0_frames, f0_frames[:, :, -1:]), 2),
            size=f0_frames.shape[-1] * self.hop_length + 1,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        f0 = f0[:, :-1, :]

        # Normalize mel and pitch
        if self.normalize:
            f0_frames = normalize(f0_frames, self.pitch_mean, self.pitch_std)
            x = normalize(x, self.mel_mean, self.mel_std)

        phase = f0_to_phase(f0, self.sample_rate)

        # noise exciter signal
        rand_noise = (
            torch.rand(
                (f0.shape[0], 1, f0.shape[1]),
                device=f0.device,
                dtype=f0.dtype,
            )
            * 2
            - 1
        )

        x = self.encoder(x)
        d_harmonic = self.decoder_harmonic(x, f0_frames, rand_noise, phase)
        d_noise = self.decoder_noise(x, f0_frames, rand_noise, phase)

        x_harmonic = self.harmonic_out(d_harmonic)
        x_noise = self.noise_out(d_noise)

        harmonic = self.harmonic(x_harmonic, f0, phase)
        noise = self.noise(x_noise, rand_noise)

        x = torch.cat((d_harmonic, d_noise), dim=1)
        if self.detach_stft:
            x = x.detach()
        x = self.merge(x)

        source = self.source(phase, rand_noise, f0)

        for layer in self.layers:
            x = layer(x, source)

        x = self.post_act(x)
        x = self.out(x)

        return x, (harmonic, noise)

    def noise(self, x, noise):
        x = torch.exp(x) / 128
        x = torch.cat((x, x[:, :, -1:]), 2)

        noise_real, noise_imag = self.stft.stft(noise)

        noise_real = noise_real * x
        noise_imag = noise_imag * x

        noise = torch.cat((noise_real, noise_imag), dim=1)

        return noise

    def harmonic(self, x, f0, phase):
        x_mag, x_phase = torch.chunk(x, 2, dim=1)
        x_mag = torch.cat((x_mag, x_mag[:, :, -1:]), 2)
        x_mag = torch.exp(x_mag) / 128

        x_phase = torch.cat((x_phase, x_phase[:, :, -1:]), 2) * torch.pi
        x_phase_r = torch.cos(x_phase)
        x_phase_i = torch.sin(x_phase)

        # combtooth exciter signal
        combtooth = torch.sinc(self.sample_rate * phase / (f0 + 1e-8))
        combtooth = combtooth.squeeze(-1)

        # harmonic part filter
        harmonic_real, harmonic_imag = self.stft.stft(combtooth)

        harmonic_real = harmonic_real * x_phase_r
        harmonic_imag = harmonic_imag * x_phase_i
        harmonic_real = harmonic_real * x_mag
        harmonic_imag = harmonic_imag * x_mag

        harmonic = torch.cat((harmonic_real, harmonic_imag), dim=1)

        return harmonic

    def source(
        self, phase: torch.Tensor, noise: torch.Tensor, f0: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            noise = noise * 0.005
            phase = 2 * torch.pi * phase

            level_harmonic = torch.arange(1, 9, device=phase.device)
            phases = phase * level_harmonic
            harmonic = torch.sin(phases).sum(-1) * 0.125
            harmonic = harmonic.unsqueeze(1)

            x = harmonic + noise

        return x

    def remove_parametrizations(self, only_nograd: bool = False) -> Self:
        param = 0
        for module in self.modules():
            if hasattr(module, "weight") and is_parametrized(module, "weight"):
                if only_nograd:
                    if module.weight.requires_grad:
                        continue
                param += 1
                remove_parametrizations(module, "weight")
        print(f"Removed {param} parametrizations.")

        return self


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        in_channels: int,
        kernel: int,
        rate: int,
        source_rate: int,
        conv_kernels: list[int],
        conv_dilations: list[int],
    ) -> None:
        super().__init__()
        self.sample_rate = config.sample_rate

        norm = get_norm(config.model.norm)

        out_channels = in_channels // 2

        self.upsample = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=rate,
            padding=(kernel - rate) // 2,
        )

        if source_rate > 1:
            self.source = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=source_rate * 2,
                stride=source_rate,
                padding=source_rate // 2,
            )
        else:
            self.source = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=1,
            )
        self.source.apply(init_weights)
        self.source = norm(self.source)

        self.convs = nn.ModuleList()
        for i in range(len(conv_kernels)):
            self.convs.append(
                ActivationBlock(
                    out_channels,
                    activation=config.model.activation,
                    kernel_size=conv_kernels[i],
                    dilation=conv_dilations,
                    snake_log=config.model.act_log,
                    upsample=config.model.act_upsample,
                    norm=config.model.norm,
                )
            )

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x_source = self.source(source)

        x = x + x_source
        xn = None
        for conv in self.convs:
            x2 = conv(x)
            if xn is None:
                xn = x2
            else:
                xn += x2

        x = xn / len(self.convs)

        return x
