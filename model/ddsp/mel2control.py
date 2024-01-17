import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .pcmer import PCmer


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Mel2Control(nn.Module):
    def __init__(self, input_channel, output_splits, layers=3):
        super().__init__()
        self.output_splits = output_splits
        self.phase_embed = nn.Linear(1, 256)
        # conv in stack
        self.stack = nn.Sequential(
            nn.Conv1d(input_channel, 256, 3, 1, 1),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
        )

        # transformer
        self.decoder = PCmer(
            num_layers=layers,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
        )
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(nn.Linear(256, self.n_out))

    def forward(self, mel, phase):
        """
        input:
            B x n_frames x n_mels
        return:
            dict of B x n_frames x feat
        """

        x = self.stack(mel.transpose(1, 2)).transpose(1, 2)
        x = x + self.phase_embed(phase / np.pi)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        return controls
