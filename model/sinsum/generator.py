from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from julius.bands import SplitBands
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from ..act import SiGLU, Swish
from ..common import f0_to_phase, normalize, remove_above_fmax
from ..layers import GRN, TransposedLayerNorm
from ..utils import get_norm, init_weights


class SinSum(nn.Module):
    def __init__(self, config, export: bool = False) -> None:
        super().__init__()
        self.export = export

        self.sample_rate = config.sample_rate
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.n_mels = config.n_mels
        self.normalize = config.norm.normalize
        self.encoder_hidden = config.model.encoder_hidden
        self.encoder_layers = config.model.encoder_layers
        self.harmonic_num = config.model.harmonic_num
        self.noise_num = config.model.noise_num
        self.max_upsample_dim = config.model.max_upsample_dim

        norm = get_norm(config.model.norm)

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

        self.register_buffer("unfold_k", torch.eye(self.hop_length)[:, None, :])
        self.register_buffer("phase_k", torch.eye(self.win_length)[:, None, :])

        self.emb = nn.Sequential(
            nn.Conv1d(self.hop_length * 2, self.encoder_hidden * 2, 3, padding=1),
            Swish(self.encoder_hidden * 2, dim=1),
            nn.Conv1d(self.encoder_hidden * 2, self.encoder_hidden, 3, padding=1),
        )
        self.emb.apply(init_weights)
        for i in range(len(self.emb)):
            if isinstance(self.emb[i], nn.Conv1d):
                self.emb[i] = norm(self.emb[i])

        self.mel = nn.Conv1d(self.n_mels, self.encoder_hidden, 1)
        self.mel.apply(init_weights)
        self.mel = norm(self.mel)

        self.f0 = nn.Conv1d(1, self.encoder_hidden, 1)
        self.f0.apply(init_weights)
        self.f0 = norm(self.f0)

        self.noise_amp = nn.ModuleList(
            [
                EncoderLayer(self.encoder_hidden, norm=config.model.norm)
                for _ in range(self.encoder_layers)
            ]
        )

        self.sin_amp = nn.ModuleList(
            [
                EncoderLayer(self.encoder_hidden, norm=config.model.norm)
                for _ in range(self.encoder_layers)
            ]
        )

        self.sin_pha = nn.ModuleList(
            [
                EncoderLayer(self.encoder_hidden, norm=config.model.norm)
                for _ in range(self.encoder_layers)
            ]
        )

        self.noise_amp_norm = TransposedLayerNorm(self.encoder_hidden)
        self.sin_amp_norm = TransposedLayerNorm(self.encoder_hidden)
        self.sin_pha_norm = TransposedLayerNorm(self.encoder_hidden)

        self.noise_amp_out = nn.Conv1d(self.encoder_hidden, self.noise_num, 1)
        self.noise_amp_out.apply(init_weights)
        self.noise_amp_out = norm(self.noise_amp_out)

        self.sin_amp_out = nn.Conv1d(self.encoder_hidden, self.harmonic_num, 1)
        self.sin_amp_out.apply(init_weights)
        self.sin_amp_out = norm(self.sin_amp_out)

        self.sin_pha_I = nn.Conv1d(self.encoder_hidden, self.harmonic_num, 1)
        self.sin_pha_I.apply(init_weights)
        self.sin_pha_I = norm(self.sin_pha_I)

        self.sin_pha_R = nn.Conv1d(self.encoder_hidden, self.harmonic_num, 1)
        self.sin_pha_R.apply(init_weights)
        self.sin_pha_R = norm(self.sin_pha_R)

        self.splitbands = SplitBands(
            sample_rate=self.sample_rate,
            cutoffs=np.linspace(0, self.sample_rate / 2, self.noise_num + 1)[1:-1],
            zeros=8,
            fft=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        f0_frames: torch.Tensor,
    ) -> torch.Tensor:
        f0 = F.interpolate(
            torch.cat((f0_frames, f0_frames[:, :, -1:]), 2),
            size=f0_frames.shape[-1] * self.hop_length + 1,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        f0 = f0[:, :-1, :]

        phase = f0_to_phase(f0, self.sample_rate)

        # sinusoid exciter signal
        sinusoid = torch.sin(2 * torch.pi * phase).transpose(1, 2)
        sinusoid_frames = F.conv1d(sinusoid, self.unfold_k, stride=self.hop_length)

        # noise exciter signal
        noise = (
            torch.rand(
                (sinusoid.shape[0], 1, sinusoid.shape[-1]),
                device=sinusoid.device,
                dtype=sinusoid.dtype,
            )
            * 2
            - 1
        )
        noise_frames = F.conv1d(noise, self.unfold_k, stride=self.hop_length)

        exciter = torch.cat([sinusoid_frames, noise_frames], dim=1)

        # Normalize mel and pitch
        if self.normalize:
            f0_frames = normalize(f0_frames, self.pitch_mean, self.pitch_std)
            x = normalize(x, self.mel_mean, self.mel_std)

        f0 = self.f0(f0_frames)
        mel = self.mel(x)
        emb = self.emb(exciter)

        x = mel + emb + f0

        noise_amp = x
        sin_amp = x
        sin_pha = x

        for layer in self.noise_amp:
            noise_amp = layer(noise_amp)
        noise_amp = self.noise_amp_norm(noise_amp)
        noise_amp: torch.Tensor = self.noise_amp_out(noise_amp)
        noise_amp = torch.exp(noise_amp) / 128

        for layer in self.sin_amp:
            sin_amp = layer(sin_amp)
        sin_amp = self.sin_amp_norm(sin_amp)
        sin_amp: torch.Tensor = self.sin_amp_out(sin_amp)
        sin_amp = torch.exp(sin_amp) / 128
        sin_amp = remove_above_fmax(sin_amp, f0_frames, float(self.sample_rate // 2), 1)

        for layer in self.sin_pha:
            sin_pha = layer(sin_pha)
        sin_pha = self.sin_pha_norm(sin_pha)
        sin_pha_I: torch.Tensor = self.sin_pha_I(sin_pha)
        sin_pha_R: torch.Tensor = self.sin_pha_R(sin_pha)

        sin_pha_IR = torch.atan2(sin_pha_I, sin_pha_R)

        level_harmonic = torch.arange(1, self.harmonic_num + 1, device=phase.device)
        harmonic = torch.zeros(
            (noise.shape[0], noise.shape[-1]),
            device=noise.device,
            dtype=noise.dtype,
        )
        phase = 2 * torch.pi * phase
        for n in range((self.harmonic_num - 1) // self.max_upsample_dim + 1):
            start = n * self.max_upsample_dim
            end = (n + 1) * self.max_upsample_dim
            phases = phase * level_harmonic[start:end]
            amplitudes = sin_amp[:, start:end, :]
            amplitudes = F.interpolate(
                torch.cat((amplitudes, amplitudes[:, :, -1:]), 2),
                size=amplitudes.shape[-1] * self.hop_length + 1,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            amplitudes = amplitudes[:, :-1, :]

            p = sin_pha_IR[:, start:end, :]
            p = F.interpolate(
                p,
                size=p.shape[-1] * self.hop_length,
                mode="nearest",
            ).transpose(1, 2)

            harmonic += (torch.sin(phases + p) * amplitudes).sum(-1)

        harmonic = harmonic.unsqueeze(1)

        noise_bands = self.splitbands(noise).squeeze(2).permute(1, 2, 0)
        noise = torch.zeros(
            (noise.shape[0], noise.shape[-1]),
            device=noise.device,
            dtype=noise.dtype,
        )

        for n in range((self.noise_num - 1) // self.max_upsample_dim + 1):
            start = n * self.max_upsample_dim
            end = (n + 1) * self.max_upsample_dim
            amplitudes = noise_amp[:, start:end, :]
            amplitudes = F.interpolate(
                torch.cat((amplitudes, amplitudes[:, :, -1:]), 2),
                size=amplitudes.shape[-1] * self.hop_length + 1,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            amplitudes = amplitudes[:, :-1, :]

            noise += (noise_bands[:, :, start:end] * amplitudes).sum(-1)

        noise = noise.unsqueeze(1)

        return harmonic + noise, (harmonic, noise)

    def remove_parametrizations(self) -> Self:
        param = 0
        for module in self.modules():
            if hasattr(module, "weight") and is_parametrized(module, "weight"):
                param += 1
                remove_parametrizations(module, "weight")
        print(f"Removed {param} parametrizations.")

        return self


class EncoderLayer(nn.Module):
    def __init__(self, hidden: int, norm: str) -> None:
        super().__init__()

        norm = get_norm(norm)

        self.conv = nn.Sequential(
            TransposedLayerNorm(hidden),
            nn.Conv1d(hidden, hidden * 4, 1),
            SiGLU(hidden * 4, dim=1),
            nn.Conv1d(hidden * 2, hidden * 2, 31, padding=15, groups=hidden * 2),
            Swish(hidden * 2, dim=1),
            GRN(hidden * 2),
            nn.Conv1d(hidden * 2, hidden, 1),
        )

        self.conv.apply(init_weights)
        for i in range(len(self.conv)):
            if isinstance(self.conv[i], nn.Conv1d):
                self.conv[i] = norm(self.conv[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(x)

        return x
