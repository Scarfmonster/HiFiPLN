import itertools

import lightning as pl
import torch
import torch.nn.functional as F
from .model import VUVEstimator
from model.hifigan.hifigan import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
    dynamic_range_compression,
)
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt
from ..utils import plot_mel, plot_x_hat
import numpy as np


class VUVTrainer(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.estimator = VUVEstimator(config)

        self.spectogram_extractor = MelSpectrogram(
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
        optim_e = torch.optim.AdamW(
            self.estimator.parameters(),
            lr=self.config.lr,
            betas=(self.config.adam_b1, self.config.adam_b2),
        )

        scheduler_e = torch.optim.lr_scheduler.ExponentialLR(
            optim_e, self.config.lr_decay
        )

        return [optim_e], [scheduler_e]

    def get_mels(self, x):
        mels = self.spectogram_extractor.to(x.device, non_blocking=True)(x.squeeze(1))
        mels = dynamic_range_compression(mels)
        return mels

    def training_step(self, batch, batch_idx):
        optim = self.optimizers()

        pitches, y, vuv = (
            batch["pitch"].float(),
            batch["audio"].float(),
            batch["vuv"].float(),
        )

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        mels = self.get_mels(y)[:, :, : mel_lens.max()]
        gen_mels = mels + torch.rand_like(mels) * self.config.model.input_noise
        vuv_hat = self.estimator(gen_mels)

        vuv = vuv[:, :, : mel_lens.max()]
        vuv_hat = vuv_hat[:, :, : mel_lens.max()]

        # Generator
        optim.zero_grad(set_to_none=True)

        loss_vuv = F.binary_cross_entropy_with_logits(vuv_hat[:, 0, :], vuv[:, 0, :])

        self.manual_backward(loss_vuv)
        optim.step()

        self.log(
            f"train_loss",
            loss_vuv,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        if self.trainer.is_last_batch:
            # Manual LR Scheduler
            scheduler_e = self.lr_schedulers()
            scheduler_e.step()

    def validation_step(self, batch, batch_idx):
        pitches, audios, vuv = (batch["pitch"], batch["audio"], batch["vuv"])

        mel_lens = batch["audio_lens"] // self.config.hop_length

        mels = self.get_mels(audios)[:, :, : mel_lens.max()]
        vuv_hat = self.estimator(mels)

        vuv = vuv[:, 0, : mel_lens.max()]
        vuv_hat = vuv_hat[:, 0, : mel_lens.max()]

        # print("VUV", vuv.shape)
        # print("HAT", vuv_hat.shape)
        # print("MELS", mels.shape)

        loss_vuv = F.binary_cross_entropy_with_logits(vuv_hat, vuv)
        # loss_vuv = F.l1_loss(F.sigmoid(vuv_hat), vuv)

        self.log(
            "valid_loss",
            loss_vuv,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        for idx, (
            v,
            gen_v,
            mel_len,
        ) in enumerate(
            zip(
                vuv.type(torch.float32).cpu().numpy(),
                F.sigmoid(vuv_hat).type(torch.float32).cpu().numpy(),
                mel_lens.cpu().numpy(),
            )
        ):
            image_vuv = plot_x_hat(v[:mel_len], gen_v[:mel_len], "VUV", "VUV HAT")

            self.logger.experiment.add_figure(
                f"sample-{idx}/vuv",
                image_vuv,
                global_step=self.global_step // 2,
            )

            plt.close(image_vuv)
