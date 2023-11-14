import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import MelSpectrogram


def plot_mel(data, titles=None):
    plt.switch_backend("agg")
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        axes[i][0].imshow(mel, origin="lower")
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


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
