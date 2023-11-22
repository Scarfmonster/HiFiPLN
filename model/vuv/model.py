import torch
import torch.nn as nn
from omegaconf import DictConfig

import numpy as np
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations, is_parametrized
import torch.nn.functional as F
from ..utils import init_weights, get_padding
from model.common import ResBlock


class VUVEstimator(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.n_mels = config.n_mels
        self.lrelu_slope = config.model_vuv.lrelu_slope
        self.hop_length = config.hop_length
        self.channels = config.model_vuv.channels

        self.res = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.conv = nn.ModuleList()
        old_ch = self.n_mels
        for x in range(config.model_vuv.layers + 1):
            new_ch = self.channels // 2**x

            self.conv.append(nn.Conv1d(old_ch, new_ch, 7, padding=3))
            self.res.append(ResBlock(new_ch, 7, lrelu_slope=self.lrelu_slope))
            self.skip.append(nn.Conv1d(self.n_mels, new_ch, 7, padding=3))

            old_ch = new_ch

        self.post = weight_norm(nn.Conv1d(new_ch, 1, 7, padding=3))

        self.conv.apply(init_weights)
        self.skip.apply(init_weights)
        self.post.apply(init_weights)

    def forward(self, x):
        org_x = x
        for c, r, s in zip(self.conv, self.res, self.skip):
            x = c(x)
            x = r(x)
            x += s(org_x)
            x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
        x = self.post(x)

        return x
