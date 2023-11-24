import itertools
from typing import Any

import lightning as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from model.hifipln.discriminator import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
)

from ..utils import STFT, plot_mel
from .generator import HiFiPLN


class HiFiPlnTrainer(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.generator = HiFiPLN(config)
        self.mpd = MultiPeriodDiscriminator(config)
        self.mrd = MultiResolutionDiscriminator(config)

        self.stft_config = config.mrd.resolutions

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
        optim_g = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=self.config.lr,
            betas=(self.config.adam_b1, self.config.adam_b2),
        )
        optim_d = torch.optim.AdamW(
            itertools.chain(self.mrd.parameters(), self.mpd.parameters()),
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

    @staticmethod
    def extract_envelope(signal, kernel_size=100, stride=50, padding=0):
        envelope = F.max_pool1d(
            signal, kernel_size=kernel_size, stride=stride, padding=padding
        )
        return envelope

    @staticmethod
    def generator_envelope_loss(y, y_hat):
        y_envelope = HiFiPlnTrainer.extract_envelope(y)
        y_hat_envelope = HiFiPlnTrainer.extract_envelope(y_hat)

        y_reverse_envelope = HiFiPlnTrainer.extract_envelope(-y)
        y_hat_reverse_envelope = HiFiPlnTrainer.extract_envelope(-y_hat)

        loss_envelope = F.l1_loss(y_envelope, y_hat_envelope) + F.l1_loss(
            y_reverse_envelope, y_hat_reverse_envelope
        )

        return loss_envelope

    def stft_loss(self, audio, gen_audio, mel_lens=None):
        loss_stft = 0.0
        for n_fft, hop_length, win_length in self.stft_config:
            audio_stft = torch.stft(
                audio.squeeze(1),
                n_fft,
                hop_length,
                win_length,
                return_complex=True,
                window=torch.hann_window(win_length, device=audio.device),
            )
            gen_audio_stft = torch.stft(
                gen_audio.squeeze(1),
                n_fft,
                hop_length,
                win_length,
                return_complex=True,
                window=torch.hann_window(win_length, device=gen_audio.device),
            )
            audio_stft = torch.view_as_real(audio_stft)
            gen_audio_stft = torch.view_as_real(gen_audio_stft)

            if mel_lens is not None:
                audio_stft = audio_stft[:, :, : mel_lens.max()]
                gen_audio_stft = gen_audio_stft[:, :, : mel_lens.max()]

            loss_stft += F.l1_loss(audio_stft, gen_audio_stft)

        return loss_stft / len(self.stft_config)

    def get_mels(self, x):
        mels = self.spectogram_extractor.get_mel(x.squeeze(1))
        return mels

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        pitches, audio = (
            batch["pitch"].float(),
            batch["audio"].float(),
        )

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        mels = self.get_mels(audio)[:, :, : mel_lens.max()]
        gen_mels = mels + torch.rand_like(mels) * self.config.model.input_noise
        gen_audio = self.generator(gen_mels, pitches)
        gen_audio_mel = self.get_mels(gen_audio)[:, :, : mel_lens.max()]

        # Generator
        optim_g.zero_grad(set_to_none=True)

        real_mpd = self.mpd(audio)
        real_mrd = self.mrd(audio)
        gen_mpd = self.mpd(gen_audio)
        gen_mrd = self.mrd(gen_audio)

        gen_loss = 0.0
        feat_loss = 0.0

        for (feat_fake, score_fake), (feat_real, _) in zip(
            gen_mpd + gen_mrd, real_mpd + real_mrd
        ):
            f_loss = 0.0
            for fake, real in zip(feat_fake, feat_real):
                f_loss += F.l1_loss(fake, real.detach())
            feat_loss += f_loss

            gen_loss += F.binary_cross_entropy_with_logits(
                score_fake, torch.ones_like(score_fake)
            )

        feat_loss = feat_loss / len(gen_mpd)

        envelope_loss = self.generator_envelope_loss(audio, gen_audio)

        stft_loss = self.stft_loss(audio, gen_audio)
        mel_loss = F.l1_loss(gen_audio_mel, mels)

        feat_loss *= 3.0
        stft_loss *= 7.5
        mel_loss *= 40.0

        loss_gen_all = gen_loss + feat_loss + envelope_loss + stft_loss + mel_loss

        self.manual_backward(loss_gen_all)
        optim_g.step()

        self.log(
            f"train/loss_gen",
            gen_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            f"train/loss_feat",
            feat_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            f"train/loss_envelope",
            envelope_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            f"train/loss_stft",
            stft_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            f"train/loss_mel",
            mel_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            f"train/loss_all",
            loss_gen_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        # Discriminator Loss
        optim_d.zero_grad(set_to_none=True)

        disc_mpd_real = self.mpd(audio)
        disc_mrd_real = self.mrd(audio)
        disc_mpd_fake = self.mpd(gen_audio.detach())
        disc_mrd_fake = self.mrd(gen_audio.detach())

        loss_d = 0.0
        for (_, score_fake), (_, score_real) in zip(
            disc_mrd_fake + disc_mpd_fake, disc_mrd_real + disc_mpd_real
        ):
            loss_d += F.binary_cross_entropy_with_logits(
                score_real, torch.ones_like(score_real)
            )
            loss_d += F.binary_cross_entropy_with_logits(
                score_fake, torch.zeros_like(score_fake)
            )

        self.manual_backward(loss_d)
        optim_d.step()

        self.log(
            f"train/loss_disc",
            loss_d,
            on_step=True,
            on_epoch=False,
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
        gen_audio = self.generator(mels, pitches)
        gen_audio_mel = self.get_mels(gen_audio)[:, :, : mel_lens.max()]

        loss_stft = self.stft_loss(audios, gen_audio, mel_lens)
        loss_mel = F.l1_loss(mels, gen_audio_mel)

        loss_valid = loss_mel + loss_stft

        self.log(
            "valid/loss_stft",
            loss_stft,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss_mel",
            loss_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss",
            loss_valid,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=pitches.shape[0],
        )

        if batch_idx == 0:
            for idx, (
                mel,
                gen_mel,
                audio,
                gen_audio,
                mel_len,
                audio_len,
            ) in enumerate(
                zip(
                    mels.cpu().numpy(),
                    gen_audio_mel.cpu().numpy(),
                    audios.cpu().type(torch.float32).numpy(),
                    gen_audio.type(torch.float32).cpu().numpy(),
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
