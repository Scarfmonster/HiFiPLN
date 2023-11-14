from typing import Any

import lightning as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..utils import STFT, plot_mel_params, plot_x_hat
from .model import PowerEstimator


class PowerTrainer(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.power_estimator = PowerEstimator(config)

        self.spectogram_extractor = STFT(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )

        self.automatic_optimization = False

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.power_estimator.parameters(),
            lr=self.config.lr,
            betas=(self.config.adam_b1, self.config.adam_b2),
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, self.config.lr_decay)

        return [optim], [scheduler]

    @staticmethod
    def extract_envelope(signal, kernel_size=100, stride=50, padding=0):
        signal = F.pad(signal, (kernel_size // 2, kernel_size // 2), mode="reflect")
        envelope = F.max_pool1d(
            signal, kernel_size=kernel_size, stride=stride, padding=padding
        )
        return envelope

    def get_mels(self, x):
        mels = self.spectogram_extractor.get_mel(x.squeeze(1))
        return mels

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()

        y = batch["audio"].float()

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        mels = self.get_mels(y)[:, :, : mel_lens.max()]
        gen_mels = mels + torch.rand_like(mels) * self.config.model.input_noise
        power_hat = self.power_estimator(gen_mels)

        power = self.extract_envelope(y, 512, 512)

        power = power[:, :, : mel_lens.max()]
        power_hat = power_hat[:, :, : mel_lens.max()]

        loss_power = F.mse_loss(power_hat[:, 0, :], power[:, 0, :])

        self.manual_backward(loss_power)
        optim.step()

        self.log(
            "train_power_loss",
            loss_power,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
            batch_size=y.shape[0],
        )

        if self.trainer.is_last_batch:
            # Manual LR Scheduler
            scheduler = self.lr_schedulers()
            scheduler.step()

    def validation_step(self, batch, batch_idx):
        audios = batch["audio"]

        mel_lens = batch["audio_lens"] // self.config.hop_length

        mels = self.get_mels(audios)[:, :, : mel_lens.max()]
        power_hat = self.power_estimator(mels)

        power = self.extract_envelope(audios, 512, 512)
        power = power[:, 0, : mel_lens.max()]
        power_hat = power_hat[:, 0, : mel_lens.max()]

        mask = (
            torch.arange(mels.shape[2], device=mels.device)[None, :] < mel_lens[:, None]
        )
        mask = mask[:, None].float()

        loss_power = F.mse_loss(power_hat * mask, power * mask)
        self.log(
            "valid_power_loss",
            loss_power,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=audios.shape[0],
        )

        if batch_idx == 0:
            for idx, (
                mel,
                power,
                gen_power,
                mel_len,
            ) in enumerate(
                zip(
                    mels.cpu().numpy(),
                    power.type(torch.float32).cpu().numpy(),
                    power_hat.type(torch.float32).cpu().numpy(),
                    mel_lens.cpu().numpy(),
                )
            ):
                image_powers = plot_x_hat(
                    power[:mel_len], gen_power[:mel_len], "POWER", "POWER HAT"
                )

                image_mel_power = plot_mel_params(
                    mel[:, :mel_len],
                    power[:mel_len],
                    gen_power[:mel_len],
                    "POWER",
                    "POWER HAT",
                )

                self.logger.experiment.add_figure(
                    f"sample-{idx}/power",
                    image_powers,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mel_power",
                    image_mel_power,
                    global_step=self.global_step,
                )

                plt.close(image_powers)
                plt.close(image_mel_power)
