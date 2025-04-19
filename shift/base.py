import abc
import numpy as np
import torch


class BaseShifter(abc.ABC):
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    @classmethod
    @abc.abstractmethod
    def shift(self, x: np.ndarray, n_steps: float) -> np.ndarray:
        raise NotImplementedError
