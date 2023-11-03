import itertools

import lightning as pl
import torch
import torch.nn.functional as F
from .hifigan import (
    HiFiGan,
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
from ..utils import plot_mel
import numpy as np


class HiFiGanTrainer(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.generator = HiFiGan(config)
        self.msd = MultiScaleDiscriminator(config)
        self.mpd = MultiPeriodDiscriminator(config)

        self.multi_scale_mels = [
            MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=0,
                f_max=config.sample_rate // 2,
                n_mels=config.n_mels,
            )
            for (n_fft, win_length, hop_length) in [
                (config.n_fft, config.win_length, config.hop_length),
                (2048, 1080, 270),
                (4096, 2160, 540),
            ]
        ]

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
        optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.lr,
            betas=(self.config.adam_b1, self.config.adam_b2),
        )
        optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            lr=self.config.lr,
            betas=(self.config.adam_b1, self.config.adam_b2),
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, self.config.lr_decay
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, self.config.lr_decay
        )

        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def generator_envelope_loss(self, y, y_hat):
        def extract_envelope(signal, kernel_size=100, stride=50):
            envelope = F.max_pool1d(signal, kernel_size=kernel_size, stride=stride)
            return envelope

        y_envelope = extract_envelope(y)
        y_hat_envelope = extract_envelope(y_hat)

        y_reverse_envelope = extract_envelope(-y)
        y_hat_reverse_envelope = extract_envelope(-y_hat)

        loss_envelope = F.l1_loss(y_envelope, y_hat_envelope) + F.l1_loss(
            y_reverse_envelope, y_hat_reverse_envelope
        )

        return loss_envelope

    def get_mels(self, x):
        mels = self.spectogram_extractor.to(x.device, non_blocking=True)(x.squeeze(1))
        mels = dynamic_range_compression(mels)
        return mels

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        pitches, y = (
            batch["pitch"].float(),
            batch["audio"].float(),
        )

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        mels = self.get_mels(y)[:, :, : mel_lens.max()]
        y_g_hat = self.generator(mels, pitches)
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : mel_lens.max()]

        # Discriminator Loss
        optim_d.zero_grad(set_to_none=True)

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        self.manual_backward(loss_disc_all)
        optim_d.step()

        self.log(
            f"train_loss_disc",
            loss_disc_all,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        # Generator
        optim_g.zero_grad(set_to_none=True)

        # We referenced STFT and Mel-Spectrogram loss from SingGAN
        # L1 STFT Loss
        stft_config = [
            (512, 50, 240),
            (1024, 120, 600),
            (2048, 240, 1200),
        ]

        loss_stft = 0
        for n_fft, hop_length, win_length in stft_config:
            y_stft = torch.stft(
                y.squeeze(1), n_fft, hop_length, win_length, return_complex=True
            )
            y_g_hat_stft = torch.stft(
                y_g_hat.squeeze(1), n_fft, hop_length, win_length, return_complex=True
            )
            y_stft = torch.view_as_real(y_stft)
            y_g_hat_stft = torch.view_as_real(y_g_hat_stft)

            loss_stft += F.l1_loss(y_stft, y_g_hat_stft)

        loss_stft /= len(stft_config)

        # L1 Mel-Spectrogram Loss
        loss_mel = 0
        for mel_transform in self.multi_scale_mels:
            mel_transform = mel_transform.to(y.device, non_blocking=True)
            y_mel = dynamic_range_compression(mel_transform(y))
            y_g_hat_mel = dynamic_range_compression(mel_transform(y_g_hat))
            # y_mel = dynamic_range_compression(y_mel)
            # y_g_hat_mel = dynamic_range_compression(y_g_hat_mel)
            loss_mel += F.l1_loss(y_mel, y_g_hat_mel)

        loss_mel /= len(self.multi_scale_mels)

        loss_aux = 0.5 * loss_stft + loss_mel

        # L1 Envelope Loss
        loss_envelope = self.generator_envelope_loss(y, y_g_hat)
        self.log(
            "train_loss_g_envelope",
            loss_envelope,
            on_step=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        # Generator Loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_all = (
            loss_gen_s
            + loss_gen_f
            + loss_fm_s
            + loss_fm_f
            + loss_envelope
            + loss_aux * 45
        )

        self.manual_backward(loss_gen_all)
        optim_g.step()

        self.log(
            f"train_loss_gen",
            loss_gen_all,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        if self.trainer.is_last_batch:
            # Manual LR Scheduler
            scheduler_g, scheduler_d = self.lr_schedulers()
            scheduler_g.step()
            scheduler_d.step()

    def validation_step(self, batch, batch_idx):
        pitches, audios = (batch["pitch"], batch["audio"])

        mel_lens = batch["audio_lens"] // self.config.hop_length

        mels = self.get_mels(audios)[:, :, : mel_lens.max()]
        y_g_hat = self.generator(mels, pitches)
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : mel_lens.max()]

        # L1 Mel-Spectrogram Loss
        # create mask
        mask = (
            torch.arange(mels.shape[2], device=mels.device)[None, :] < mel_lens[:, None]
        )
        mask = mask[:, None].float()

        loss_mel = F.l1_loss(mels * mask, y_g_hat_mel * mask)
        self.log(
            "valid_loss",
            loss_mel,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        for idx, (mel, gen_mel, audio, gen_audio, mel_len, audio_len) in enumerate(
            zip(
                mels.cpu().numpy(),
                y_g_hat_mel.cpu().numpy(),
                audios.cpu().type(torch.float32).numpy(),
                y_g_hat.type(torch.float32).cpu().numpy(),
                mel_lens.cpu().numpy(),
                batch["audio_lens"].cpu().numpy(),
            )
        ):
            image_mels = plot_mel(
                [
                    gen_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
            )

            self.logger.experiment.add_figure(
                f"sample-{idx}/mels",
                image_mels,
                global_step=self.global_step // 2,
            )
            self.logger.experiment.add_audio(
                f"sample-{idx}/wavs/gt",
                audio[0, :audio_len],
                self.global_step // 2,
                sample_rate=self.config.sample_rate,
            )
            self.logger.experiment.add_audio(
                f"sample-{idx}/wavs/prediction",
                gen_audio[0, :audio_len],
                self.global_step // 2,
                sample_rate=self.config.sample_rate,
            )

            plt.close(image_mels)
