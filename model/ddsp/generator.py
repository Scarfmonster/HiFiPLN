import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from .mel2control import Mel2Control
from ..stft import STFT


class DDSP(nn.Module):
    """
    DDSP (Differentiable Digital Signal Processing) module.

    Args:
        config (DictConfig): Configuration dictionary.
    """

    def __init__(self, config: DictConfig, layers=3, inputs=None) -> None:
        super().__init__()

        self.inputs = inputs if inputs is not None else config.n_mels

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

        self.mel2ctrl = Mel2Control(self.inputs, split_map, layers=layers)

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
        mel_frames = mel_frames.transpose(-1, -2)
        f0 = F.interpolate(
            torch.cat((f0_frames, f0_frames[:, :, -1:]), 2),
            size=f0_frames.shape[-1] * self.hop_length + 1,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        f0 = f0[:, :-1, :]
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

    def remove_parametrizations(self):
        """
        Remove parametrizations from the module.
        """
        param = 0
        for module in self.modules():
            if hasattr(module, "weight") and is_parametrized(module, "weight"):
                param += 1
                remove_parametrizations(module, "weight")
        print(f"Removed {param} parametrizations.")
