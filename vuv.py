import librosa
import numpy as np
import pyworld
import torch
from omegaconf import DictConfig
from torchaudio.functional import highpass_biquad, lowpass_biquad


class VUVEstimator:
    def __init__(self, config: DictConfig) -> None:
        self.sample_rate = config.sample_rate
        self.win_length = config.win_length // 4
        self.hop_length = config.hop_length // 4
        self.f0_min = config.preprocessing.f0_min
        self.f0_max = config.preprocessing.f0_max
        self.f_max = config.f_max
        self.vuv_smoothing = config.preprocessing.vuv_smoothing
        self.zcr_uv = 0.25
        self.zcr_v = 0.03
        self.rms_uv = 0.01

    def get_vuv(self, audio, f0):
        audio = highpass_biquad(audio, self.sample_rate, self.f0_min)
        audio = lowpass_biquad(audio, self.sample_rate, self.f_max)

        max_loudness = torch.max(torch.abs(audio))
        if max_loudness > 0:
            audio /= max_loudness

        ap = self.get_world(audio, f0)
        ap = ap[:, 0]

        audio = audio.cpu().numpy()[0].astype(np.float64)

        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            threshold=0.001,
        )
        zcr = zcr[0]
        zcr = np.convolve(zcr, np.hanning(7) / 3, "same")

        rms = self.get_rms(
            audio, win_length=self.win_length, hop_length=self.hop_length
        )
        rms = rms[0]

        vuv = 1 - (np.ones_like(ap) * (ap > 0.01))

        for i in range(len(vuv)):
            if zcr[i] > self.zcr_uv:
                vuv[i] = 0
            elif zcr[i] < self.zcr_v and rms[i] > self.rms_uv:
                vuv[i] = 1
            elif rms[i] <= self.rms_uv:
                vuv[i] = 0

        vuv = np.convolve(
            vuv,
            np.hanning(self.vuv_smoothing) / (self.vuv_smoothing / 2),
            "same",
        )
        vuv = np.interp(
            np.linspace(0, np.max(vuv), len(vuv) // 4),
            np.linspace(0, np.max(vuv), len(vuv)),
            vuv,
        )

        vuv = np.ones_like(vuv) * (vuv >= 0.5)

        for s in range(1, self.vuv_smoothing + 1):
            self.smooth(vuv, s)

        return vuv.astype(np.float32)

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
        org_len = len(arr)
        arr = np.pad(arr, s, "reflect")
        for i in range(s - 1, org_len - s):
            m = np.mean(np.concatenate((arr[i - s : i], arr[i + 1 : i + s + 1])))
            if m > 0.5:
                arr[i] = 1
            elif m < 0.5:
                arr[i] = 0
        return arr[s:-s]
