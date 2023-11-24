import torch
import torch.nn.functional as F

from data import VocoderDataset
from model.hifigan.hifigan import dynamic_range_compression
from model.hifipln.generator import HiFiPLN
from omegaconf import OmegaConf

# from torchaudio.transforms import MelSpectrogram
from model.utils import plot_mel, STFT
import soundfile

config = OmegaConf.load("configs/hifipln.yaml")

valid_dataset = VocoderDataset(config, "train")
valid_dataset.segment_length = None
valid_dataset.loudness_shift = None

spectogram_extractor = STFT(
    sample_rate=config.sample_rate,
    n_fft=config.n_fft,
    win_length=config.win_length,
    hop_length=config.hop_length,
    f_min=config.f_min,
    f_max=config.f_max,
    n_mels=config.n_mels,
)


def get_mels(x):
    mels = spectogram_extractor.get_mel(x.squeeze(1))
    return mels


def power(x, win_length):
    x = torch.exp(x)
    x = x**2

    power = 2 * torch.sum(x, axis=-2, keepdims=True) / win_length**2
    power = torch.sqrt(power)

    power = torch.clamp(power, 0.0, 1.0)

    return power


all_max = 0.0
all_min = 1.0
maxes = []
mins = []

for i, item in enumerate(valid_dataset):
    audio = torch.tensor(item["audio"]).cuda()
    max_loudness = torch.max(torch.abs(audio))
    if max_loudness > 0:
        audio /= max_loudness
    mel = get_mels(audio)
    p = power(mel, config.n_mels)
    mel_max = torch.max(p).cpu()
    mel_min = torch.max(p).cpu()
    maxes.append(mel_max)
    mins.append(mel_min)

    if i % 10000 == 0:
        print("Processing", i)

    if mel_max > all_max:
        print("Max:", mel_max)
        all_max = mel_max

    if mel_min < all_min:
        print("Min:", mel_min)
        all_min = mel_min

maxes = torch.tensor(maxes)
print()
print("Max:", all_max)
print("Min:", all_min)
print("Mean:", torch.mean(maxes))
print("Median:", torch.median(maxes))
