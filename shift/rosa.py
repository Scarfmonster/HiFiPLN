from .base import BaseShifter
import numpy as np
import librosa


class RosaShifter(BaseShifter):
    def __init__(self, sample_rate: int = 44100) -> None:
        super().__init__(sample_rate)

    def shift(self, x: np.ndarray, n_steps: float) -> np.ndarray:
        x = librosa.effects.pitch_shift(
            x,
            sr=self.sample_rate,
            n_steps=n_steps,
            scale=True,
        )
        return x
