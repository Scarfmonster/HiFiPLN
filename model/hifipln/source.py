import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class NoiseCombSource(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.sample_rate = config.sample_rate
        self.n_mels = config.n_mels
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.window = torch.hann_window(self.win_length)

        self.magnitudes = nn.Sequential(
            nn.Conv1d(self.n_mels, 128, 7, 1, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 128, 3, 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 2, 3, 1, padding=1),
            nn.Hardtanh(0, 1, inplace=True),
        )

    def forward(self, mel_frames, f0_frames):
        if f0_frames.ndim == 2:
            f0_frames = f0_frames[:, None]
        f0 = F.interpolate(
            f0_frames,
            scale_factor=self.hop_length,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        x = torch.cumsum(f0.double() / self.sample_rate, axis=1)
        x = x - torch.round(x)
        x = x.to(f0)

        # combtooth exciter signal
        combtooth = torch.sinc(self.sample_rate * x / (f0 + 1e-8))
        combtooth = combtooth.squeeze(-1)

        # noise exciter signal
        noise = torch.randn_like(combtooth)

        # magnitudes
        magnitudes = self.magnitudes(mel_frames)

        magnitudes = F.interpolate(
            magnitudes,
            scale_factor=self.hop_length,
            mode="linear",
            align_corners=True,
        )

        noise_magnitudes = magnitudes[:, 0, :]
        combtooth_magnitudes = magnitudes[:, 1, :]

        # noise
        noise = noise * noise_magnitudes
        combtooth = combtooth * combtooth_magnitudes

        noise = F.hardtanh(noise, -1, 1)
        combtooth = F.hardtanh(combtooth, -1, 1)

        signal = noise + combtooth

        signal = F.hardtanh(signal, -1, 1)

        return signal.unsqueeze(-2), (combtooth.unsqueeze(-2), noise.unsqueeze(-2))
