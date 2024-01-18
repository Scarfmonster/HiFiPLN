import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.utils.parametrizations import weight_norm

from ..ddsp.pcmer import Attention
from ..utils import init_weights


class PreEncoder(nn.Module):
    def __init__(self, config: DictConfig, outputs: int) -> None:
        super().__init__()

        self.n_mels = config.n_mels
        self.upsample_initial = config.model.upsample_initial

        self.convs1 = nn.Sequential(
            weight_norm(nn.Conv1d(self.n_mels, 256, 3, padding=1)),
            nn.GroupNorm(4, 256),
            nn.GELU(),
        )

        self.norm1 = nn.LayerNorm(256)
        self.attention1 = Attention(256, heads=8, dim_head=32)
        self.dropout1 = nn.Dropout(0.2)

        self.convs2 = nn.Sequential(
            weight_norm(nn.Conv1d(256 + 1, 512, 3, padding=1)),
            nn.GLU(1),
            weight_norm(nn.Conv1d(256, 256, 3, padding=1)),
            nn.GELU(),
        )
        self.norm2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(0.2)

        self.post = weight_norm(nn.Linear(256, outputs))

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x, f0):
        x = self.convs1(x)

        x = x.transpose(1, 2)
        a = self.attention1(x)
        a = self.dropout1(a)
        x = x + a
        x = self.norm1(x)
        x = x.transpose(1, 2)

        c = self.convs2(torch.cat([x, f0], dim=1))
        x = x + c
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.dropout2(x)

        x = self.post(x)
        x = x.transpose(1, 2)

        return x
