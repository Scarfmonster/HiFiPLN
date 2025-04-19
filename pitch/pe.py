import abc

import librosa
import numpy as np
import parselmouth
import pyreaper
import pyworld
import torch

from .rmvpe import RMVPE


class BasePE(abc.ABC):
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        f0_min: int = 0,
        f0_max: int = 22050,
        keep_zeros: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros

    def __call__(
        self, x: torch.Tensor, pad_to: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.process(x)

        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float()

        if pad_to is None:
            pad_to = f0.shape[0]
        else:
            total_pad = pad_to - f0.shape[0]
            if total_pad > 0:
                f0 = np.pad(
                    f0, (total_pad // 2, total_pad - total_pad // 2), "constant"
                )
                f0 = torch.from_numpy(f0).float()
            elif total_pad < 0:
                f0 = f0[:total_pad]

        vuv = torch.ones_like(f0)
        vuv *= f0 > 0

        if self.keep_zeros:
            return f0, vuv, f0

        org_f0 = torch.clone(f0)

        # Remove zero frequencies and linearly interpolate
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex)
        time_org = self.hop_length / self.sample_rate * nzindex
        time_frame = (
            torch.arange(pad_to, device=x.device) * self.hop_length / self.sample_rate
        )

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device), vuv, org_f0

        if f0.shape[0] == 1:
            return (
                torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0],
                vuv,
                org_f0,
            )

        return (
            self.interpolate(time_frame, time_org, f0, left=f0[0], right=f0[-1]),
            vuv,
            org_f0,
        )

    def interpolate(
        self,
        x: torch.Tensor,
        xp: torch.Tensor,
        fp: torch.Tensor,
        left: torch.Tensor | None = None,
        right: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Interpolate a 1-D function.

        Args:
            x (torch.Tensor): The x-coordinates at which to evaluate the interpolated values.
            xp (torch.Tensor): A 1-D array of monotonically increasing real values.
            fp (torch.Tensor): A 1-D array of real values, same length as xp.
            left (torch.Tensor, optional): Value to return for x < xp[0], default is fp[0].
            right (torch.Tensor, optional): Value to return for x > xp[-1], default is fp[-1].

        Returns:
            torch.Tensor: The interpolated values, same shape as x.
        """

        # Ref: https://github.com/pytorch/pytorch/issues/1552#issuecomment-979998307
        i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
        interped = (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (
            xp[i] - xp[i - 1]
        )

        if left is None:
            left = fp[0]

        interped = torch.where(x < xp[0], left, interped)

        if right is None:
            right = fp[-1]

        interped = torch.where(x > xp[-1], right, interped)

        return interped

    @classmethod
    @abc.abstractmethod
    def process(self, x: torch.Tensor) -> torch.Tensor | np.ndarray:
        raise NotImplementedError


class RmvpePE(BasePE):
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        f0_min: int = 40,
        f0_max: int = 1100,
        keep_zeros: bool = True,
    ) -> None:
        super().__init__(sample_rate, hop_length, f0_min, f0_max, keep_zeros=keep_zeros)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros

        self.rmvpe = RMVPE("ckpt/rmvpe.pt")

    def process(self, x: torch.Tensor) -> np.ndarray:
        f0_out = None
        chunk_length = 600 * self.sample_rate
        chunk_length = chunk_length - (chunk_length % self.hop_length)
        chunks = torch.split(x, chunk_length, dim=-1)
        for c in chunks:
            length = c.shape[-1] // self.hop_length
            f0, uv = self.rmvpe.get_pitch(
                c, self.sample_rate, length, hop_size=self.hop_length
            )
            if f0_out is None:
                f0_out = f0
            else:
                f0_out = np.concatenate((f0_out, f0), axis=0)

        return f0_out


class ReaperPE(BasePE):
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        f0_min: int = 40,
        f0_max: int = 1100,
        keep_zeros: bool = True,
    ) -> None:
        super().__init__(sample_rate, hop_length, f0_min, f0_max, keep_zeros=keep_zeros)

    def process(self, x: torch.Tensor) -> np.ndarray:
        x2 = x.cpu().numpy()[0].astype(np.float64)
        x2 = (x2 * 32768).astype(np.int16)

        pm_times, pm, f0_times, f0, corr = pyreaper.reaper(
            x2,
            fs=self.sample_rate,
            minf0=self.f0_min,
            maxf0=self.f0_max,
            frame_period=self.hop_length / self.sample_rate,
        )

        # Replace -1 values with 0
        f0[f0 < 0] = 0.0

        return f0


class HarvestPE(BasePE):
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        f0_min: int = 40,
        f0_max: int = 1100,
        keep_zeros: bool = True,
    ) -> None:
        super().__init__(sample_rate, hop_length, f0_min, f0_max, keep_zeros=keep_zeros)

    def process(self, x: torch.Tensor) -> np.ndarray:
        x2 = x.cpu().numpy()[0].astype(np.float64)

        _f0, t = pyworld.harvest(
            x2,
            self.sample_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.hop_length / self.sample_rate * 1000,
        )

        f0 = pyworld.stonemask(x2, _f0, t, self.sample_rate)

        return f0


class DioPE(BasePE):
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        f0_min: int = 40,
        f0_max: int = 1100,
        keep_zeros: bool = True,
    ) -> None:
        super().__init__(sample_rate, hop_length, f0_min, f0_max, keep_zeros=keep_zeros)

    def process(self, x: torch.Tensor) -> np.ndarray:
        x2 = x.cpu().numpy()[0].astype(np.float64)

        _f0, t = pyworld.dio(
            x2,
            self.sample_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.hop_length / self.sample_rate * 1000,
            allowed_range=0.11,
        )

        f0 = pyworld.stonemask(x2, _f0, t, self.sample_rate)

        return f0


class ParselmouthPE(BasePE):
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        f0_min: int = 40,
        f0_max: int = 1100,
        keep_zeros: bool = True,
        very_accurate: bool = True,
    ) -> None:
        super().__init__(sample_rate, hop_length, f0_min, f0_max, keep_zeros=keep_zeros)
        self.very_accurate = very_accurate

    def process(self, x: torch.Tensor) -> np.ndarray:
        x2 = x.cpu().numpy()[0].astype(np.float64)

        if self.very_accurate:
            pad = 3.0
        else:
            pad = 1.5

        l_pad = int(np.ceil(pad / self.f0_min * self.sample_rate))
        r_pad = (
            self.hop_length * ((len(x2) - 1) // self.hop_length + 1)
            - len(x2)
            + l_pad
            + 1
        )
        x2 = np.pad(x2, (l_pad, r_pad))

        s = parselmouth.Sound(x2, sampling_frequency=self.sample_rate).to_pitch_ac(
            time_step=self.hop_length / self.sample_rate,
            voicing_threshold=0.55,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max,
            very_accurate=self.very_accurate,
            voiced_unvoiced_cost=0.1554,
        )
        assert np.abs(s.t1 - pad / self.f0_min) < 0.001
        f0 = s.selected_array["frequency"].astype(np.float32)
        return f0


class PyinPE(BasePE):
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        f0_min: int = 40,
        f0_max: int = 1100,
        keep_zeros: bool = True,
    ) -> None:
        super().__init__(sample_rate, hop_length, f0_min, f0_max, keep_zeros=keep_zeros)

    def process(self, x: torch.Tensor) -> np.ndarray:
        x2 = x.cpu().numpy()[0].astype(np.float64)

        f0, voiced_flag, voiced_prob = librosa.pyin(
            x2,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            frame_length=self.hop_length * 4,
            win_length=self.hop_length * 2,
            hop_length=self.hop_length,
            fill_na=0.0,
            center=True,
            pad_mode="reflect",
        )

        return f0


class MixedPE(BasePE):
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        f0_min: int = 40,
        f0_max: int = 1100,
        keep_zeros: bool = True,
    ) -> None:
        super().__init__(sample_rate, hop_length, f0_min, f0_max, keep_zeros)
        self.harvest = HarvestPE(sample_rate, hop_length, f0_min, f0_max, keep_zeros)
        self.parsel = ParselmouthPE(sample_rate, hop_length, f0_min, f0_max, keep_zeros)
        self.dio = DioPE(sample_rate, hop_length, f0_min, f0_max, keep_zeros)

    def process(self, x: torch.Tensor) -> np.ndarray:
        f0p, _, _ = self.parsel(x)
        f0h, _, _ = self.harvest(x, pad_to=f0p.shape[-1])
        f0y, _, _ = self.dio(x, pad_to=f0p.shape[-1])

        f0 = []

        for f in zip(f0h, f0p, f0y):
            f = sorted(f)

            d1 = f[1] - f[0]
            d2 = f[2] - f[1]

            if d1 < d2:
                f0.append((f[0] + f[1]) / 2)
            elif d2 < d1:
                f0.append((f[1] + f[2]) / 2)
            else:
                f0.append(f[1])

        return np.array(f0)
