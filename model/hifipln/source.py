import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from .stft import STFT


class DDSP(nn.Module):
    """
    DDSP (Differentiable Digital Signal Processing) module.

    Args:
        config (DictConfig): Configuration dictionary.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.sample_rate = config.sample_rate
        self.n_mels = config.n_mels
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.window = torch.hann_window(self.win_length)

        self.stft = STFT(
            self.n_fft, self.hop_length, self.win_length, self.window, center=True
        )

        # Mel2Control
        split_map = {
            "harmonic_magnitude": self.win_length // 2 + 1,
            "harmonic_phase": self.win_length // 2 + 1,
            "noise_magnitude": self.win_length // 2 + 1,
        }

        self.mel2ctrl = Encoder(self.n_mels, split_map)

    def forward(self, mel_frames, f0_frames, max_upsample_dim=32):
        """
        Forward pass of the DDSP module.

        Args:
            mel_frames (torch.Tensor): Mel frames of shape B x n_mels x n_frames.
            f0_frames (torch.Tensor): F0 frames of shape B x 1 x n_frames.
            max_upsample_dim (int): Maximum upsample dimension.

        Returns:
            torch.Tensor: The generated signal.
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the harmonic and noise components.
        """
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

        phase = 2 * torch.pi * x
        phase_frames = phase[:, :: self.hop_length, :]

        # parameter prediction
        ctrls = self.mel2ctrl(mel_frames, phase_frames)

        harmonic = self.combsub(f0, x, ctrls)

        noise_param = torch.exp(ctrls["noise_magnitude"]) / 128
        noise_param = torch.cat((noise_param, noise_param[:, -1:, :]), 1)
        noise_param = noise_param.permute(0, 2, 1)

        # noise part filter
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise_real, noise_imag = self.stft.stft(noise)

        noise_real = noise_real * noise_param
        noise_imag = noise_imag * noise_param

        noise = self.stft.istft(noise_real, noise_imag, harmonic.shape[-1])

        harmonic = F.hardtanh(harmonic, min_val=-1, max_val=1)
        noise = F.hardtanh(noise, min_val=-1, max_val=1)

        signal = harmonic + noise
        signal = F.hardtanh(signal, min_val=-1, max_val=1)

        return signal.unsqueeze(-2), (
            harmonic.unsqueeze(-2),
            noise.unsqueeze(-2),
        )

    def combsub(self, f0, x, ctrls):
        """
        The harmonic part of the signal.

        Args:
            f0 (torch.Tensor): F0 tensor.
            x (torch.Tensor): Cumulative sum of F0 tensor.
            ctrls (Dict[str, torch.Tensor]): Control parameters.

        Returns:
            torch.Tensor: The harmonic part of the signal.
        """
        src_allpass = ctrls["harmonic_phase"]
        src_allpass = torch.cat((src_allpass, src_allpass[:, -1:, :]), 1)
        src_allpass = src_allpass.permute(0, 2, 1)
        src_param = torch.exp(ctrls["harmonic_magnitude"]) / 128
        src_param = torch.cat((src_param, src_param[:, -1:, :]), 1)
        src_param = src_param.permute(0, 2, 1)

        # combtooth exciter signal
        combtooth = torch.sinc(self.sample_rate * x / (f0 + 1e-8))
        combtooth = combtooth.squeeze(-1)

        # harmonic part filter
        harmonic_real, harmonic_imag = self.stft.stft(combtooth)
        mags = torch.sqrt(harmonic_real**2 + harmonic_imag**2)
        phase = torch.atan2(harmonic_imag, harmonic_real)

        mags = mags * src_param
        phase = phase + src_allpass
        phase = phase.clamp(-torch.pi, torch.pi)

        harmonic_real = mags * torch.cos(phase)
        harmonic_imag = mags * torch.sin(phase)
        harmonic = self.stft.istft(harmonic_real, harmonic_imag, combtooth.shape[-1])

        return harmonic


class Encoder(nn.Module):
    def __init__(self, inputs: int, output_splits) -> None:
        super().__init__()

        self.output_splits = output_splits

        self.input = nn.Sequential(
            nn.Conv1d(inputs, 256, 3, padding=1),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
        )

        self.phase = nn.Linear(1, 256)

        self.encoder = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.GLU(1),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.GLU(1),
            nn.Conv1d(256, 256, 3, padding=1),
        )

        self.norm = nn.LayerNorm(256)

        self.n_out = sum([v for k, v in output_splits.items()])
        self.output = nn.Linear(256, self.n_out)

    def forward(self, x, phase):
        x = self.input(x).transpose(1, 2)
        x = x + self.phase(phase / torch.pi)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)

        x = self.norm(x)
        x = self.output(x)

        x = split_to_dict(x, self.output_splits)

        return x


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))
