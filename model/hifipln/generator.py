import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from model.common import ResBlock
from model.power.model import PowerEstimator
from model.vuv.model import VUVEstimator

from ..utils import init_weights


class HiFiPLN(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.hop_length = config.hop_length
        self.lrelu_slope = config.model.lrelu_slope
        self.upsample_rates = config.model.upsample_rates
        self.upsample_kernel_sizes = config.model.upsample_kernel_sizes
        self.resblock_kernel_sizes = config.model.resblock_kernel_sizes
        self.resblock_dilation_sizes = config.model.resblock_dilation_sizes
        self.upsample_initial_channel = config.model.upsample_initial_channel
        self.num_upsamples = len(config.model.upsample_rates)
        self.num_kernels = len(config.model.resblock_kernel_sizes)
        self.n_mels = config.n_mels

        self.source = SourceNoise(config)
        self.pre_conv = weight_norm(
            nn.Conv1d(self.n_mels, self.upsample_initial_channel, 7, 1, padding=3)
        )
        self.noise_convs = nn.ModuleList()

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                self.upsample_rates,
                self.upsample_kernel_sizes,
            )
        ):
            c_cur = self.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        self.upsample_initial_channel // (2**i),
                        self.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(self.upsample_rates):
                stride_f0 = np.prod(self.upsample_rates[i + 1 :])
                self.noise_convs.append(
                    weight_norm(
                        nn.Conv1d(
                            1,
                            c_cur,
                            kernel_size=stride_f0 * 2,
                            stride=stride_f0,
                            padding=stride_f0 // 2,
                        )
                    )
                )
            else:
                self.noise_convs.append(weight_norm(nn.Conv1d(1, c_cur, kernel_size=1)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    self.resblock_kernel_sizes,
                    self.resblock_dilation_sizes,
                )
            ):
                self.resblocks.append(ResBlock(ch, k, d, self.lrelu_slope))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.vuv = VUVEstimator(config)
        self.load_vuv(config.model.vuv_ckpt)
        self.power = PowerEstimator(config)
        self.load_power(config.model.power_ckpt)

    def forward(self, x, f0):
        if f0.ndim == 2:
            f0 = f0[:, None]

        f0 = F.interpolate(
            f0, size=x.shape[-1] * self.hop_length, mode="linear"
        ).transpose(1, 2)

        with torch.no_grad():
            vuv = self.vuv(x)
            power = self.power(x)

        source = self.source(f0, power, vuv)
        source = source.transpose(1, 2)

        x = self.pre_conv(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](source)
            x = x + x_source
            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        x = F.leaky_relu(x, inplace=True)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x, vuv, power

    def load_vuv(self, ckpt_path):
        cp_dict = torch.load(ckpt_path, map_location="cpu")

        self.vuv.load_state_dict(
            {
                k.replace("estimator.", ""): v
                for k, v in cp_dict["state_dict"].items()
                if k.startswith("estimator.")
            }
        )

        for param in self.vuv.parameters():
            param.requires_grad = False

    def load_power(self, ckpt_path):
        cp_dict = torch.load(ckpt_path, map_location="cpu")

        self.power.load_state_dict(
            {
                k.replace("power_estimator.", ""): v
                for k, v in cp_dict["state_dict"].items()
                if k.startswith("power_estimator.")
            }
        )

        for param in self.power.parameters():
            param.requires_grad = False

    def remove_parametrizations(self):
        param = 0
        for module in self.modules():
            if hasattr(module, "weight") and is_parametrized(module, "weight"):
                param += 1
                remove_parametrizations(module, "weight")
        print(f"Removed {param} parametrizations.")


class SourceNoise(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.noise_amp = config.model.noise_amp
        self.sine_amp = config.model.sine_amp
        self.lrelu_slope = config.model.lrelu_slope
        self.harmonic_num = config.model.harmonic_num

        self.sines = SineGen(config)
        self.linear = nn.Linear(self.harmonic_num + 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, f0, power, vuv):
        sines = self.linear(self.sines(f0))
        noise = torch.rand_like(sines)

        vuv = F.sigmoid(vuv)

        vuv_amp = F.interpolate(vuv, size=sines.shape[-2], mode="linear").transpose(
            1, 2
        )
        power_amp = F.interpolate(power, size=sines.shape[-2], mode="linear").transpose(
            1, 2
        )

        sines *= vuv_amp
        noise_amp = vuv_amp * self.noise_amp + (1 - vuv_amp) * self.sine_amp
        noise *= noise_amp

        source = sines + noise

        source = source * power_amp

        return self.tanh(source)


class SineGen(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.sample_rate = config.sample_rate
        self.sine_amp = config.model.sine_amp
        self.harmonic_num = config.model.harmonic_num
        self.dim = self.harmonic_num + 1

    def _f02sine(self, f0):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0 / self.sample_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0.shape[0], f0.shape[2], device=f0.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)

        # To prevent torch.cumsum numerical overflow,
        # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
        # Buffer tmp_over_one_idx indicates the time step to add -1.
        # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp

        return sine_waves
