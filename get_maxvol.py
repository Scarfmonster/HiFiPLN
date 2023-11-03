import torch
import torch.nn.functional as F

from data import VocoderDataset
from model.hifigan.hifigan import dynamic_range_compression
from model.hifipln.generator import HiFiPLN
from omegaconf import OmegaConf
from torchaudio.transforms import MelSpectrogram
from model.utils import plot_mel
import soundfile

config = OmegaConf.load("configs/hifipln.yaml")

valid_dataset = VocoderDataset(config, "train")
valid_dataset.segment_length = None
valid_dataset.loudness_shift = None

spectogram_extractor = MelSpectrogram(
    sample_rate=config.sample_rate,
    n_fft=config.n_fft,
    win_length=config.win_length,
    hop_length=config.hop_length,
    f_min=config.f_min,
    f_max=config.f_max,
    n_mels=config.n_mels,
).to("cuda:0", non_blocking=True)


def get_mels(x):
    mels = spectogram_extractor(x.squeeze(1))
    mels = dynamic_range_compression(mels)
    return mels


all_max = 0.0
maxes = []

for i, item in enumerate(valid_dataset):
    audio = torch.tensor(item["audio"]).cuda()
    mel = get_mels(audio)
    mel = torch.exp(mel)
    # mel = mel[:, 0:64, :]
    # mel_amp = mel.transpose(2, 1)
    # mel_amp = F.max_pool1d(torch.exp(mel_amp), 128, 1)
    mel_amp = torch.sum(mel, 1)
    mel_amp = mel_amp.squeeze()
    mel_max = torch.max(torch.abs(mel_amp))
    mel_max = mel_max.cpu()
    maxes.append(mel_max)

    if i % 10000 == 0:
        print("Processing", i)

    if mel_max > all_max:
        print(mel_max)
        all_max = mel_max

maxes = torch.tensor(maxes)
print("Max:", all_max)
print("Mean:", torch.mean(maxes))
print("Median:", torch.median(maxes))
