import importlib
import os

import h5py
import numpy as np
import pyworld
import torch
from lightning import LightningDataModule
from natsort import os_sorted
from omegaconf import DictConfig
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import MelSpectrogram
from torchdata.stateful_dataloader import StatefulDataLoader


class VocoderDataModule(LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

    def train_dataloader(self) -> StatefulDataLoader:
        if not hasattr(self, "_train_dataset"):
            self._train_dataset = VocoderDataset(self.config, "train")
        if not hasattr(self, "_train_dataloader"):
            self._train_dataloader = StatefulDataLoader(
                self._train_dataset,
                batch_size=self.config.dataloader.train.batch_size,
                shuffle=self.config.dataloader.train.shuffle,
                num_workers=self.config.dataloader.train.num_workers,
                pin_memory=self.config.dataloader.train.pin_memory,
                drop_last=self.config.dataloader.train.drop_last,
                persistent_workers=self.config.dataloader.train.persistent_workers,
                prefetch_factor=self.config.dataloader.train.prefetch_factor,
                collate_fn=collate_fn,
            )
        return self._train_dataloader

    def val_dataloader(self) -> StatefulDataLoader:
        if not hasattr(self, "_val_dataset"):
            self._val_dataset = VocoderDataset(self.config, "valid")
        if not hasattr(self, "_val_dataloader"):
            self._val_dataloader = StatefulDataLoader(
                self._val_dataset,
                batch_size=self.config.dataloader.valid.batch_size,
                shuffle=self.config.dataloader.valid.shuffle,
                num_workers=self.config.dataloader.valid.num_workers,
                pin_memory=self.config.dataloader.valid.pin_memory,
                drop_last=self.config.dataloader.valid.drop_last,
                persistent_workers=self.config.dataloader.valid.persistent_workers,
                prefetch_factor=self.config.dataloader.valid.prefetch_factor,
                collate_fn=collate_fn,
            )
        return self._val_dataloader

    def state_dict(self) -> dict:
        return {
            "train_dataloader": self.train_dataloader().state_dict(),
            "val_dataloader": self.val_dataloader().state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.train_dataloader().load_state_dict(state_dict["train_dataloader"])
        print("Restored train dataloader state")

        self.val_dataloader().load_state_dict(state_dict["val_dataloader"])
        print("Restored val dataloader state")


class VocoderDataset(Dataset):
    def __init__(self, config: DictConfig, split: str) -> None:
        super().__init__()
        self.split = split
        self.path = config.dataset[split].path
        if os.path.isdir(self.path):
            self.mode = "numpy"
            self.items = self.get_items()
        elif os.path.isfile(self.path) and self.path.endswith(".h5"):
            self.mode = "hdf5"
            self.dataset = h5py.File(self.path, "r")
            self.audio = self.dataset["audio"]
            self.pitch = self.dataset["pitch"]
            self.vuv = self.dataset["vuv"]
        else:
            raise ValueError(f"Invalid dataset path: {self.path}")

        self.segment_length = config.dataset[split].segment_length
        self.sample_rate = config.sample_rate
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft
        self.pitch_shift = config.dataset[split].pitch_shift
        self.pitch_shift_prob = config.dataset[split].pitch_shift_prob
        self.loudness_shift = config.dataset[split].loudness_shift
        self.loudness_shift_prob = config.dataset[split].loudness_shift_prob
        self.hap_shift = config.dataset[split].hap_shift
        self.hap_shift_prob = config.dataset[split].hap_shift_prob
        self.reverse_prob = config.dataset[split].reverse_prob
        self.target_pitch = config.dataset[split].get("target_pitch", True)
        self.target_loudness = config.dataset[split].get("target_loudness", True)
        self.target_hap = config.dataset[split].get("target_hap", True)
        self.target_reverse = config.dataset[split].get("target_reverse", True)
        self.return_vuv = config.dataset[split].get("return_vuv", False)
        self.separate_sp_ap = config.dataset[split].get("separate_sp_ap", False)
        self.shift_f0 = config.dataset[split].get("shift_f0", True)
        self.highpass = config.dataset[split].get("highpass", True)
        self.f_max = config.f_max
        self.f_min = config.f_min
        self.pitch_f_min = config.preprocessing.f0_min
        self.pitch_f_max = config.preprocessing.f0_max

        if self.hap_shift and not self.separate_sp_ap:
            raise ValueError("Harmonic-Aperiodic shift requires separate_sp_ap")

        self.spectogram_extractor = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )

        shifter = config.dataset[split].get("pitch_shifter", "RosaShifter")
        pitch_shifter_cls = getattr(__import__("shift", fromlist=[shifter]), shifter)
        self.pitch_shifter = pitch_shifter_cls(sample_rate=config.sample_rate)

    def __len__(self) -> int:
        if self.mode == "numpy":
            return len(self.items)
        elif self.mode == "hdf5":
            return self.audio.shape[0]

    def __del__(self):
        if self.mode == "hdf5":
            self.dataset.close()

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return self.get_item(idx)

    def get_item(self, idx):
        if self.mode == "numpy":
            x = np.load(self.items[idx], allow_pickle=True).item()
            audio = torch.from_numpy(x["audio"]).float()
            pitch = torch.from_numpy(x["pitch"]).float()
            vuv = torch.from_numpy(x["vuv"]).float() if "vuv" in x else None
            harmonic = (
                torch.from_numpy(x["harmonic"]).float() if "harmonc" in x else None
            )
            aperiodic = (
                torch.from_numpy(x["aperiodic"]).float() if "aperiodic" in x else None
            )
        elif self.mode == "hdf5":

            pitch = torch.from_numpy(self.pitch[idx]).float()
            vuv = (
                torch.from_numpy(self.vuv[idx]).float()
                if self.vuv is not None
                else None
            )
            audio = self.audio[idx]
            if audio.ndim == 2:
                harmonic = torch.from_numpy(audio[1]).float()
                aperiodic = torch.from_numpy(audio[2]).float()
                audio = audio[0]
            else:
                harmonic = None
                aperiodic = None

            audio = torch.from_numpy(audio).float()

        # Change loudness
        max_loudness = torch.max(torch.abs(audio))

        if max_loudness > 1.0:
            audio = audio / max_loudness

        audio_org = audio.clone()
        audio_target = audio.clone()

        pitch = pitch[None, None, :]
        vuv = vuv[None, None, :]

        if self.separate_sp_ap:
            if harmonic is None or aperiodic is None:
                harmonic, aperiodic = self.separate_audio(audio, pitch[0, 0])

        if self.hap_shift and np.random.random_sample() < self.hap_shift_prob:
            ratio = (
                np.random.random_sample() * (self.hap_shift[1] - self.hap_shift[0])
                + self.hap_shift[0]
            )
            harmonic = harmonic * min(1, 1 + ratio)
            aperiodic = aperiodic * min(1, 1 - ratio)

            audio = harmonic + aperiodic
            if self.target_hap:
                audio_target = audio.clone()

        if (
            self.pitch_shift is not None
            and np.random.random_sample() < self.pitch_shift_prob
        ):
            pitch_steps = (
                np.random.random_sample() * (self.pitch_shift[1] - self.pitch_shift[0])
                + self.pitch_shift[0]
            )
            if pitch_steps != 0:
                pitch_scale = 2 ** (pitch_steps / 12)

                audio = audio.numpy().astype(np.float64)
                audio = self.pitch_shifter.shift(
                    audio,
                    n_steps=pitch_steps,
                )
                audio = torch.from_numpy(audio).float()

                if self.target_pitch:
                    audio_target = audio_target.numpy().astype(np.float64)
                    audio_target = self.pitch_shifter.shift(
                        audio_target,
                        n_steps=pitch_steps,
                    )
                    audio_target = torch.from_numpy(audio_target).float()

                if self.separate_sp_ap:
                    harmonic = harmonic.numpy().astype(np.float64)
                    harmonic = self.pitch_shifter.shift(
                        harmonic,
                        n_steps=pitch_steps,
                    )
                    harmonic = torch.from_numpy(harmonic).float()

                    aperiodic = aperiodic.numpy().astype(np.float64)
                    aperiodic = self.pitch_shifter.shift(
                        aperiodic,
                        n_steps=pitch_steps,
                    )
                    aperiodic = torch.from_numpy(aperiodic).float()

                assert audio.shape[-1] == audio_org.shape[-1]

                if self.shift_f0:
                    pitch *= pitch_scale

        if self.segment_length and audio.shape[-1] > self.segment_length:
            audio_length = audio.shape[-1]
            start = np.random.randint(0, audio.shape[-1] - self.segment_length + 1)

            if audio.shape[-1] == audio_org.shape[-1]:
                audio_org = audio_org[start : start + self.segment_length]
            else:
                start_org = np.random.randint(
                    0, audio_org.shape[-1] - self.segment_length + 1
                )
                audio_org = audio_org[start_org : start_org + self.segment_length]

            audio = audio[start : start + self.segment_length]
            audio_target = audio_target[start : start + self.segment_length]

            if self.separate_sp_ap:
                harmonic = harmonic[start : start + self.segment_length]
                aperiodic = aperiodic[start : start + self.segment_length]

            pitch = interpolate(pitch, audio_length, mode="linear", align_corners=True)
            pitch = pitch[:, :, start : start + self.segment_length]

            if self.return_vuv and vuv is not None:
                vuv = interpolate(
                    vuv,
                    size=audio_length,
                    mode="linear",
                    align_corners=True,
                )
                vuv = vuv[:, :, start : start + self.segment_length]

        if self.highpass:
            audio = highpass_biquad(audio, self.sample_rate, self.f_min)
            audio_org = highpass_biquad(audio_org, self.sample_rate, self.f_min)
            audio_target = highpass_biquad(audio_target, self.sample_rate, self.f_min)
            if self.separate_sp_ap:
                harmonic = highpass_biquad(harmonic, self.sample_rate, self.f_min)
                aperiodic = highpass_biquad(aperiodic, self.sample_rate, self.f_min)

        if pitch.shape[-1] != audio.shape[-1] // self.hop_length:
            pitch = interpolate(
                pitch,
                audio.shape[-1] // self.hop_length,
                mode="linear",
                align_corners=True,
            )

        pitch = pitch[0, 0, :]

        if self.return_vuv and vuv is not None:
            if vuv.shape[-1] != audio.shape[-1] // self.hop_length:
                vuv = interpolate(
                    vuv,
                    audio.shape[-1] // self.hop_length,
                    mode="linear",
                    align_corners=True,
                )[0, 0, :]
                vuv = torch.where(vuv > 0.5, 1.0, 0.0)
            else:
                vuv = vuv[0, 0, :]

        max_loudness = torch.max(torch.abs(audio))
        if (
            self.loudness_shift is not None
            and max_loudness > 0.0
            and np.random.random_sample() < self.loudness_shift_prob
        ):
            audio /= max_loudness
            if self.separate_sp_ap:
                harmonic /= max_loudness
                aperiodic /= max_loudness

            loudness_factor = (
                np.random.random_sample()
                * (self.loudness_shift[1] - self.loudness_shift[0])
                + self.loudness_shift[0]
            )
            max_loudness = max_loudness.cpu().item()
            if loudness_factor > max_loudness * 3:
                loudness_factor = max_loudness * 3

            audio *= loudness_factor
            if self.separate_sp_ap:
                harmonic *= loudness_factor
                aperiodic *= loudness_factor
            if self.target_loudness:
                audio_target /= max_loudness
                audio_target *= loudness_factor

        else:
            loudness_factor = 1.0

        audio = audio[None]
        audio_target = audio_target[None]
        audio_org = audio_org[None]
        pitch = pitch[None]
        loudness_factor = torch.tensor(loudness_factor)[None]
        data = {
            "audio": audio,
            "audio_target": audio_target,
            "audio_org": audio_org,
            "pitch": pitch,
            "loudness": loudness_factor,
        }

        if self.separate_sp_ap:
            data["harmonic"] = harmonic[None]
            data["aperiodic"] = aperiodic[None]

        if self.return_vuv and vuv is not None:
            data["vuv"] = vuv[None]

        if (
            self.reverse_prob is not None
            and np.random.random_sample() < self.reverse_prob
        ):
            for k in data:
                if k in ("loudness",):
                    continue
                data[k] = torch.flip(data[k], [1])

        return data

    def get_items(self) -> list[str]:
        items = []
        for dirpath, _, filenames in os.walk(self.path):
            for f in filenames:
                if f.endswith(".npy"):
                    items.append(os.path.join(dirpath, f))

        items = os_sorted(items)

        return items

    # Adapted from
    def separate_audio(self, audio, f0):
        audio = audio.numpy()
        f0 = f0.numpy()

        audio = audio.astype(np.double) + np.random.randn(*audio.shape) * 1e-5
        f0 = f0.astype(np.double)

        wav_frames = (audio.shape[0] + self.hop_length - 1) // self.hop_length
        f0_frames = f0.shape[0]
        if f0_frames < wav_frames:
            f0 = np.pad(
                f0,
                (0, wav_frames - f0_frames),
                mode="constant",
                constant_values=(f0[0], f0[-1]),
            )
        elif f0_frames > wav_frames:
            f0 = f0[:wav_frames]

        time_step = self.hop_length / self.sample_rate
        t = np.arange(0, wav_frames) * time_step
        sp = pyworld.cheaptrick(audio, f0, t, self.sample_rate, fft_size=self.n_fft)
        ap = pyworld.d4c(audio, f0, t, self.sample_rate, fft_size=self.n_fft)

        harmonic = pyworld.synthesize(
            f0,
            np.clip(sp * (1 - ap * ap), a_min=1e-16, a_max=None),  # clip to avoid zeros
            np.zeros_like(ap),
            self.sample_rate,
            frame_period=time_step * 1000,
        )
        harmonic = torch.from_numpy(harmonic).float()

        aperiodic = pyworld.synthesize(
            f0,
            sp * ap * ap,
            np.ones_like(ap),
            self.sample_rate,
            frame_period=time_step * 1000,
        )
        aperiodic = torch.from_numpy(aperiodic).float()

        return harmonic, aperiodic


def collate_fn(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    all_keys = set(j for i in data for j in i.keys())
    data = {k: [i[k] for i in data] for k in all_keys}

    for k in all_keys:
        stacked, lens, max_len = pad_and_stack(data[k])
        data[k] = stacked
        data[k + "_lens"] = lens
        data[k + "_max_len"] = max_len

    return data


def pad_and_stack(
    x: list[torch.Tensor] | list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
