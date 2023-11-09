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
        self.lrelu_slope = config.model.lrelu_slope
        self.hop_length = config.hop_length

        self.pre = weight_norm(nn.Conv1d(self.n_mels, 512, 3, padding=1))
        self.res1 = ResBlock(512, 3)
        self.conv1 = weight_norm(nn.Conv1d(512, 256, 3, padding=1))
        self.skip1 = weight_norm(nn.Conv1d(self.n_mels, 256, 1))
        self.res2 = ResBlock(256, 3)
        self.conv2 = weight_norm(nn.Conv1d(256, 128, 3, padding=1))
        self.skip2 = weight_norm(nn.Conv1d(self.n_mels, 128, 1))
        self.res3 = ResBlock(128, 3)
        self.conv3 = weight_norm(nn.Conv1d(128, 64, 3, padding=1))
        self.skip3 = weight_norm(nn.Conv1d(self.n_mels, 64, 3, padding=1))
        self.res4 = ResBlock(64, 3)
        self.post = weight_norm(nn.Conv1d(64, 1, 3, padding=1))
        self.apply(init_weights)

    def forward(self, x):
        s1 = self.skip1(x)
        s2 = self.skip2(x)
        s3 = self.skip3(x)
        x = self.pre(x)
        x = self.res1(x)
        x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
        x = self.conv1(x)
        x = x + s1
        x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
        x = self.res2(x)
        x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
        x = self.conv2(x)
        x = x + s2
        x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
        x = self.res3(x)
        x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
        x = self.conv3(x)
        x = x + s3
        x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
        x = self.res4(x)
        x = self.post(x)

        return x
