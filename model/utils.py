from collections import deque
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn import Module
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.nn.utils.parametrize import is_parametrized
from torchaudio.transforms import MelSpectrogram


def plot_mel(data, titles=None):
    plt.switch_backend("agg")
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    fig.set_figwidth(12)
    fig.set_figheight(2.5 * len(data))
    if titles is None:
        titles = [None for i in range(len(data))]
    plt.tight_layout()

    # Make sure that all mels are displayed with the same scale
    v_min = np.log(1e-5)
    v_max = np.log(1e-5)
    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        m = np.max(mel)
        if m > v_max:
            v_max = m

    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        axes[i][0].imshow(
            mel,
            origin="lower",
            vmin=v_min,
            vmax=v_max,
            interpolation="bilinear",
            interpolation_stage="rgba",
        )
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def plot_x_hat(vuv, vuv_hat, label1, label2):
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.plot(vuv, label=label1)
    ax.plot(vuv_hat, label=label2)
    ax.legend(loc="upper right")

    return fig


def plot_mel_params(mel, vuv, vuv_hat, label1, label2):
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.imshow(mel, origin="lower")
    ax.set_aspect(2.5, adjustable="box")
    ax.set_ylim(0, mel.shape[0])
    ax.plot(vuv * mel.shape[0], label=label1)
    ax.plot(vuv_hat * mel.shape[0], label=label2)
    ax.legend(loc="upper right")

    return fig


def get_mel_transform(
    sample_rate=44100,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    f_min=40,
    f_max=16000,
    n_mels=128,
    center=True,
    power=1.0,
    pad_mode="reflect",
    norm="slaney",
    mel_scale="slaney",
) -> torch.Tensor:
    transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        center=center,
        power=power,
        pad_mode=pad_mode,
        norm=norm,
        mel_scale=mel_scale,
        window_fn=torch.hann_window,
        normalized=False,
    )

    return transform


def plot_snakes(model: torch.nn.Module, logscale=False) -> None | plt.Figure:
    alphas = []
    betas = []
    gammas = []
    for layer in model.modules():
        classname = layer.__class__.__name__
        if classname in ("Snake", "SnakeBeta", "SnakeGamma", "ReSnake"):
            alphas.append(layer.alpha.detach().cpu().numpy())
            if hasattr(layer, "beta"):
                betas.append(layer.beta.detach().cpu().numpy())
            if hasattr(layer, "gamma"):
                gammas.append(layer.gamma.detach().cpu().numpy())

    if len(alphas) == 0 and len(betas) == 0 and len(gammas) == 0:
        return None

    if logscale:
        for i in range(len(alphas)):
            alphas[i] = np.exp(alphas[i])
        for i in range(len(betas)):
            betas[i] = np.exp(betas[i])
        for i in range(len(gammas)):
            gammas[i] = np.exp(gammas[i])
    else:
        for i in range(len(alphas)):
            alphas[i] += 1.0
        for i in range(len(betas)):
            betas[i] += 1.0
        for i in range(len(gammas)):
            gammas[i] += 1.0

    subplots = 1

    if len(betas) > 0:
        subplots += 1
        for i in range(len(betas)):
            betas[i] = 1 / (betas[i] + 1e-8)

    if len(gammas) > 0:
        subplots += 1

    plt.switch_backend("agg")
    fig, ax = plt.subplots(subplots)
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(
        top=0.96, bottom=0.04, left=0.04, right=0.98, hspace=0.1, wspace=0.0
    )
    fig.set_figwidth(12)
    fig.set_figheight(2.5 * subplots)

    if subplots == 1:
        ax = [ax]

    ax[0].eventplot(alphas, orientation="vertical")
    ax[0].set_title("Alpha", fontsize="medium")
    ax[0].tick_params(labelsize="x-small")

    if len(betas) > 0:
        ax[1].eventplot(betas, orientation="vertical")
        ax[1].set_title("Beta", fontsize="medium")
        ax[1].tick_params(labelsize="x-small")

    if len(gammas) > 0:
        ax[2].eventplot(gammas, orientation="vertical")
        ax[2].set_title("Gamma", fontsize="medium")
        ax[2].tick_params(labelsize="x-small")

    return fig


def plot_weights(model: torch.nn.Module):
    weights = []
    for layer in model.modules():
        classname = layer.__class__.__name__
        if classname in (
            "Conv1d",
            "ConvTranspose1d",
            "ParametrizedConv1d",
            "ParametrizedConvTranspose1d",
        ):
            if hasattr(layer, "weight"):
                if (
                    not is_parametrized(layer, "weight")
                    and not layer.weight.requires_grad
                ):
                    continue
                weight = torch.flatten(layer.weight)

                weights.append(weight.detach().cpu().numpy())

    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    plt.tight_layout()
    fig.set_figwidth(12)

    ax.eventplot(weights, orientation="vertical")
    ax.set_title("Weights", fontsize="medium")
    ax.tick_params(labelsize="x-small")

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        m.weight.data.normal_(mean, std)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_norm(norm: str) -> Callable[[Module], Module]:
    match norm:
        case "weight":
            return weight_norm
        case "spectral":
            return spectral_norm
        case "none":
            return noop_norm
        case _:
            raise ValueError(f"Unknown norm type: {norm}")


class STFT:
    def __init__(
        self,
        sample_rate=44100,
        n_mels=128,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        f_min=40,
        f_max=16000,
        clip_val=1e-5,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.fmin = f_min
        self.fmax = f_max
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = torch.hann_window(win_length)

    def get_mel(self, y: torch.Tensor, center: bool = False) -> torch.Tensor:
        mel_basis_key = str(self.fmax) + "_" + str(y.device)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_size - self.hop_length) // 2,
                (self.win_size - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self.hann_window.to(y.device),
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()

        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression(spec, clip_val=self.clip_val)
        # spec *= 0.434294

        return spec


def dynamic_range_compression(
    x: torch.Tensor,
    C: float = 1.0,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def noop_norm(module: Module, **kwargs):
    return module


class AutoClip:
    def __init__(
        self,
        percentile: float = 10.0,
        history_size: int = 1000,
        max_grad: float | None = 100.0,
    ) -> None:
        self.percentile: float = percentile
        self.history_size: int = history_size
        self.history = np.empty(history_size, dtype=np.float32)
        self.history[:] = np.nan
        self.tail: int = -1
        self.max_grad: float | None = max_grad

    def __call__(self, x: torch.nn.Module) -> float:
        if self.tail == -1:
            if self.max_grad is None:
                clip_val = 100.0
            else:
                clip_val = self.max_grad
        else:
            clip_val = np.nanpercentile(self.history, self.percentile)

        if self.max_grad is not None:
            clip_val = min(clip_val, self.max_grad)
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(x.parameters(), clip_val)
            .detach()
            .cpu()
            .numpy()
        )

        self.tail = (self.tail + 1) if self.tail < self.history_size - 1 else 0
        self.history[self.tail] = grad_norm

        return clip_val

    def state_dict(self):
        return {"history": self.history.tolist(), "tail": self.tail}

    def load_state_dict(self, state_dict):
        self.history = np.array(state_dict["history"], dtype=np.float32)
        self.tail = state_dict["tail"]
