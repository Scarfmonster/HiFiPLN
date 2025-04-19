import librosa
import numpy as np
import pyworld
import torch
from omegaconf import DictConfig
from torchaudio.functional import highpass_biquad, lowpass_biquad


class VUVEstimator:
    def __init__(self, config: DictConfig) -> None:
        self.sample_rate = config.sample_rate
        self.oversampling = config.preprocessing.oversampling
        self.win_length = config.win_length
        self.hop_length = config.hop_length // self.oversampling
        self.f0_min = config.preprocessing.f0_min
        self.f0_max = config.preprocessing.f0_max
        self.f_max = config.f_max
        self.vuv_smoothing = config.preprocessing.vuv_smoothing

    def get_vuv(self, audio, f0):
        ap = self.get_world(audio, f0)
        ap = ap[:, 0]

        vuv = 1 - (np.ones_like(ap) * (ap > 0.5))

        vuv = self.clean_vuv(vuv)

        return vuv

    def clean_vuv(self, vuv):
        if self.vuv_smoothing > 1:
            for i in range(2, self.vuv_smoothing + 1):
                vuv = self.smooth(vuv, i)

        if self.oversampling > 1:
            vuv = np.interp(
                np.linspace(0, np.max(vuv), len(vuv) // self.oversampling),
                np.linspace(0, np.max(vuv), len(vuv)),
                vuv,
            )

        return vuv

    def from_f0(self, f0):
        vuv = f0 > 0
        vuv = self.clean_vuv(vuv)

        return vuv

    def get_world(self, audio, f0):
        time_step = self.hop_length / self.sample_rate
        wav_frames = (audio.shape[-1] + self.hop_length - 1) // self.hop_length
        t = np.arange(0, wav_frames) * time_step

        f0 = f0.cpu().numpy().astype(np.float64)

        if f0.shape[0] < wav_frames - 1:
            f0 = np.pad(
                f0,
                (0, wav_frames - f0.shape[0]),
                mode="constant",
                constant_values=(f0[0], f0[-1]),
            )
        elif f0.shape[0] > wav_frames - 1:
            f0 = f0[:wav_frames]
        ap = pyworld.d4c(
            audio.cpu().numpy().astype(np.float64)[0],
            f0,
            t,
            self.sample_rate,
            fft_size=self.hop_length * 4,
        )

        return ap

    def get_rms(self, audio, win_length=2048, hop_length=512):
        S = librosa.magphase(
            librosa.stft(
                audio,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
                center=True,
                pad_mode="reflect",
            )
        )[0]
        rms = librosa.feature.rms(S=S)

        return rms

    @staticmethod
    def smooth(arr, s):
        arr = np.pad(arr, s // 2, "reflect")
        arr = np.convolve(arr, np.ones(s), mode="valid") / s
        arr = np.where(arr > 0.5, 1, 0)
        return arr
