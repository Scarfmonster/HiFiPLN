import torch
import torch.nn as nn


class Exp(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


def f0_to_phase(f0: torch.Tensor, sample_rate: int) -> torch.Tensor:
    phase = torch.cumsum(f0.double() / sample_rate, dim=1)
    phase -= torch.round(phase)
    phase = phase.to(dtype=f0.dtype)

    return phase


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def remove_above_fmax(
    amplitudes: torch.Tensor,
    pitch: torch.Tensor,
    fmax: float,
    level_start: int = 1,
) -> torch.Tensor:
    amplitudes = amplitudes.transpose(1, 2)
    pitch = pitch.transpose(1, 2)
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(level_start, n_harm + level_start).to(pitch)
    aa = (pitches < fmax).float() + 1e-7
    amplitudes = amplitudes * aa
    return amplitudes.transpose(1, 2)


def leaky_clamp(x: torch.Tensor, slope: float = 0.01) -> torch.Tensor:
    x = torch.where(x < 0.0, x * slope, x)
    x = torch.where(x >= 1.0, 1.0 - slope + x * slope, x)

    return x


def noise_dropout(
    mels: torch.Tensor,
    noise: float,
    dropout: float,
    mel_mean: torch.Tensor | None = None,
    mel_std: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise == 0.0 and dropout == 0.0:
        return mels
    mels = mels.clone()

    if noise > 0.0:
        if mel_mean is None or mel_std is None:
            std, mean = torch.std_mean(mels, dim=-1, keepdim=True)
        else:
            std, mean = mel_std, mel_mean
            std = std.expand_as(mels)
            mean = mean.expand_as(mels)
        noise_scale = (
            torch.rand(mels.shape[0], 1, 1, device=mels.device, dtype=mels.dtype)
            * noise
        )
        mels += torch.normal(mean, std) * noise_scale

    if dropout > 0.0:
        for b in range(mels.shape[0]):
            if dropout > torch.rand(1):
                # Drop a random spectogram row
                row = torch.randint(0, mels.shape[1], (1,))
                mels[b, row, :] = 0

    return mels


def atan2(y, x):
    # Create a pi tensor with the same device and data type as y
    pi = torch.tensor(torch.pi, device=y.device, dtype=y.dtype)
    half_pi = pi / 2
    eps = 1e-7

    x += -0.0
    y += -0.0

    near_zeros = x.abs() < eps
    x = x * (near_zeros.logical_not())
    x = x + (near_zeros * x.sign() * eps)

    # Compute the arctangent of y/x
    ans = y / x
    a = torch.where(ans > 1, 1 / ans, ans)
    a = torch.where(ans < -1, 1 / a, a)
    aa = a * a
    r = 0.0
    r = r * aa + 0.00289394245323327
    r = r * aa - 0.0162911733512761
    r = r * aa + 0.0431408641542157
    r = r * aa - 0.0755120841589429
    r = r * aa + 0.10668127080775
    r = r * aa - 0.142123340834229
    r = r * aa + 0.199940412794435
    r = r * aa - 0.333331728467737
    r = r * aa + 1.0
    a = r * a
    a = torch.where(ans > 1, half_pi - a, a)
    a = torch.where(ans < -1, -half_pi - a, a)
    ans = a

    # Create boolean tensors representing positive, negative, and zero values of y and x
    y_positive = y > 0
    y_negative = y < 0
    x_positive = x >= 0
    x_negative = x < 0
    x_zero = x.abs() < eps
    y_zero = y.abs() < eps

    zeros = torch.zeros_like(ans)

    # Adjust ans based on the positive, negative, and zero values of y and x
    ans += torch.where(y_positive & x_negative, pi, zeros)  # Quadrants I and II
    ans -= torch.where(y_negative & x_negative, pi, zeros)  # Quadrants III and IV
    ans = torch.where(y_positive & x_zero, half_pi, ans)  # Positive y-axis
    ans = torch.where(y_negative & x_zero, -half_pi, ans)  # Negative y-axis
    ans = torch.where(y_zero & x_negative, pi, ans)
    ans = torch.where(y_zero & x_positive, zeros, ans)

    ans = torch.nan_to_num(ans, nan=0.0, posinf=torch.pi, neginf=-torch.pi)

    return ans


def sinc(x: torch.Tensor) -> torch.Tensor:
    piX = torch.pi * x
    return torch.where(
        x == 0, torch.tensor(1, device=x.device, dtype=x.dtype), torch.sin(piX) / piX
    )
