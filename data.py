from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import Dataset
import os
import numpy as np
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import resample, highpass_biquad
from torch.nn.functional import interpolate
import torch


class VocoderDataset(Dataset):
    def __init__(self, config: DictConfig, split: str) -> None:
        super().__init__()
        self.split = split
        self.path = config.dataset[split].path
        self.items = self.get_items()

        self.segment_length = config.dataset[split].segment_length
        self.sample_rate = config.sample_rate
        self.hop_length = config.hop_length
        self.window_length = config.win_length
        self.pitch_shift = config.dataset[split].pitch_shift
        self.loudness_shift = config.dataset[split].loudness_shift
        self.return_vuv = config.dataset[split].get("return_vuv", False)
        self.f_min = config.f_min

        self.spectogram_extractor = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)

    def get_item(self, idx):
        x = np.load(self.items[idx], allow_pickle=True).item()
        audio = torch.from_numpy(x["audio"]).float()
        pitch = torch.from_numpy(x["pitch"]).float()
        vuv = torch.from_numpy(x["vuv"]).float() if "vuv" in x else None

        audio = highpass_biquad(audio, self.sample_rate, self.f_min)

        # Change loudness
        max_loudness = torch.max(torch.abs(audio))

        if max_loudness > 1.0:
            audio = audio / max_loudness

        if self.pitch_shift is not None:
            pitch_steps = np.random.randint(
                self.pitch_shift[0], self.pitch_shift[1] + 1
            )
            if pitch_steps != 0:
                duration_shift = 2 ** (pitch_steps / 12)
                orig_sr = round(self.sample_rate * duration_shift)
                orig_sr = orig_sr - (orig_sr % 100)

                audio = resample(
                    audio,
                    orig_freq=orig_sr,
                    new_freq=self.sample_rate,
                )

                pitch *= 2 ** (pitch_steps / 12)

        pitch = interpolate(
            pitch[None, None, :], audio.shape[-1], mode="linear", align_corners=True
        )[0, 0, :]

        if self.segment_length and audio.shape[-1] > self.segment_length:
            audio_length = audio.shape[-1]
            start = np.random.randint(0, audio.shape[-1] - self.segment_length + 1)
            audio = audio[start : start + self.segment_length]
            pitch = pitch[start : start + self.segment_length]

            if self.return_vuv and vuv is not None:
                vuv = interpolate(
                    vuv[None, None, :], audio_length, mode="linear", align_corners=True
                )
                vuv = vuv[:, :, start : start + self.segment_length]
                vuv = interpolate(
                    vuv,
                    self.segment_length // self.hop_length,
                    mode="linear",
                    align_corners=True,
                )[0, 0, :]
                vuv = torch.where(vuv > 0.5, 1, 0)

        max_loudness = torch.max(torch.abs(audio))
        if max_loudness > 0:
            audio /= max_loudness

        if self.loudness_shift is not None:
            factor = (
                np.random.random_sample()
                * (self.loudness_shift[1] - self.loudness_shift[0])
                + self.loudness_shift[0]
            )
            audio *= factor

        audio = audio[None]
        pitch = pitch[None]
        data = {"audio": audio, "pitch": pitch}

        if self.return_vuv and vuv is not None:
            data["vuv"] = vuv[None]
        return data

    def get_items(self):
        items = []
        for dirpath, _, filenames in os.walk(self.path):
            for f in filenames:
                if f.endswith(".npy"):
                    items.append(os.path.join(dirpath, f))

        return items


def collate_fn(data):
    all_keys = set(j for i in data for j in i.keys())
    data = {k: [i[k] for i in data] for k in all_keys}

    for k in all_keys:
        stacked, lens, max_len = pad_and_stack(data[k])
        data[k] = stacked
        data[k + "_lens"] = lens
        data[k + "_max_len"] = max_len

    return data


def pad_and_stack(x):
    dim = -1
    if isinstance(x[0], np.ndarray):
        x = [torch.from_numpy(i).float() for i in x]

    lens = torch.LongTensor([i.shape[dim] for i in x])
    max_len = torch.max(lens)

    if dim < 0:
        pads = (0,) * (abs(dim + 1) * 2)
    else:
        negative_pad_dim = dim - len(x[0].shape) + 1
        pads = (0,) * (abs(negative_pad_dim) * 2)

    stacked = torch.stack(
        [torch.nn.functional.pad(i, pads + (0, max_len - i.shape[dim])) for i in x]
    )

    return (
        stacked,
        lens,
        max_len,
    )
