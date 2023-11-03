import itertools
from typing import Any

import lightning as pl
import torch
import torch.nn.functional as F
from .model import PowerEstimator
from model.hifigan.hifigan import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
    dynamic_range_compression,
)
from omegaconf import DictConfig

# from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt
from ..utils import plot_mel, plot_x_hat, plot_mel_params, get_mel_transform


class PowerTrainer(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.power_estimator = PowerEstimator(config)

        self.spectogram_extractor = get_mel_transform(
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
        envelope = F.max_pool1d(
            signal, kernel_size=kernel_size, stride=stride, padding=padding
        )
        return envelope

    @staticmethod
    def generator_envelope_loss(y, y_hat):
        y_envelope = PowerEstimator.extract_envelope(y)
        y_hat_envelope = PowerEstimator.extract_envelope(y_hat)

        y_reverse_envelope = PowerEstimator.extract_envelope(-y)
        y_hat_reverse_envelope = PowerEstimator.extract_envelope(-y_hat)

        loss_envelope = F.l1_loss(y_envelope, y_hat_envelope) + F.l1_loss(
            y_reverse_envelope, y_hat_reverse_envelope
        )

        return loss_envelope

    def get_mels(self, x):
        mels = self.spectogram_extractor.to(x.device, non_blocking=True)(x.squeeze(1))
        mels = dynamic_range_compression(mels)
        return mels

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()

        y = batch["audio"].float()

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        mels = self.get_mels(y)[:, :, : mel_lens.max()]
        gen_mels = mels + torch.rand_like(mels) * self.config.model.input_noise
        power_hat = self.power_estimator(gen_mels)

        power = self.extract_envelope(y, 2048, 512, 1024)
        power = power[:, :, : mel_lens.max()]
        power_hat = power_hat[:, :, : mel_lens.max()]

        loss_power = F.mse_loss(power_hat[:, 0, :], power[:, 0, :])

        self.manual_backward(loss_power)
        optim.step()

        self.log(
            "train_power_loss",
            loss_power,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
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

        power = self.extract_envelope(audios, 1024, 512, 256)
        power = power[:, 0, : mel_lens.max()]
        power_hat = power_hat[:, 0, : mel_lens.max()]

        mask = (
            torch.arange(mels.shape[2], device=mels.device)[None, :] < mel_lens[:, None]
        )
        mask = mask[:, None].float()

        loss_power = F.mse_loss(power_hat * mask, power * mask)
        # loss_power = F.binary_cross_entropy_with_logits(power_hat * mask, power * mask)
        self.log(
            "valid_power_loss",
            loss_power,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=audios.shape[0],
        )

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
                global_step=self.global_step // 2,
            )
            self.logger.experiment.add_figure(
                f"sample-{idx}/mel_power",
                image_mel_power,
                global_step=self.global_step // 2,
            )

            plt.close(image_powers)
            plt.close(image_mel_power)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pitches, audios = (batch["pitch"].float(), batch["audio"].float())
        mel_lens = batch["audio_lens"] // self.config.hop_length
        mels = self.get_mels(audios)[:, :, : mel_lens.max()]

        return self.power_estimator(mels, pitches)
