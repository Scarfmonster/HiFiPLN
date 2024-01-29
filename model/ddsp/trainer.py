import itertools

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
from .generator import DDSP
from .loss import MSSLoss, RSSLoss, UVLoss


class DDSPTrainer(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.generator = DDSP(config)
        if self.config.model_ddsp.use_discriminator:
            self.mpd = MultiPeriodDiscriminator(config)
            self.mrd = MultiResolutionDiscriminator(config)
        else:
            self.ddsp_loss = RSSLoss(256, 2048, 4)

        self.uv_loss = UVLoss(
            config.hop_length, uv_tolerance=config.model_ddsp.uv_tolerance
        )

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

        self.mss_loss = MSSLoss([2048, 1024, 512, 256, 128])

        self.automatic_optimization = False

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=self.config.lr,
            betas=(self.config.adam_b1, self.config.adam_b2),
            fused=True,
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, self.config.lr_decay
        )

        if self.config.model_ddsp.use_discriminator:
            optim_d = torch.optim.AdamW(
                itertools.chain(self.mrd.parameters(), self.mpd.parameters()),
                lr=self.config.lr,
                betas=(self.config.adam_b1, self.config.adam_b2),
                fused=True,
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                optim_d, self.config.lr_decay
            )
            return [optim_g, optim_d], [scheduler_g, scheduler_d]

        return [
            optim_g,
        ], [
            scheduler_g,
        ]

    @staticmethod
    def extract_envelope(signal, kernel_size=100, stride=50, padding=0):
        envelope = F.max_pool1d(
            signal, kernel_size=kernel_size, stride=stride, padding=padding
        )
        return envelope

    @staticmethod
    def envelope_loss(y, y_hat):
        y_envelope = DDSPTrainer.extract_envelope(y)
        y_hat_envelope = DDSPTrainer.extract_envelope(y_hat)

        y_reverse_envelope = DDSPTrainer.extract_envelope(-y)
        y_hat_reverse_envelope = DDSPTrainer.extract_envelope(-y_hat)

        loss_envelope = F.l1_loss(y_envelope, y_hat_envelope) + F.l1_loss(
            y_reverse_envelope, y_hat_reverse_envelope
        )

        return loss_envelope

    def get_mels(self, x):
        mels = self.spectogram_extractor.get_mel(x.squeeze(1))
        return mels

    def training_step(self, batch, batch_idx):
        if self.config.model_ddsp.use_discriminator:
            optim_g, optim_d = self.optimizers()
            current_step = self.global_step // 2
        else:
            optim_g = self.optimizers()
            current_step = self.global_step

        pitches, audio, vuv = (
            batch["pitch"].float(),
            batch["audio"].float(),
            batch["vuv"].float(),
        )

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        mels = self.get_mels(audio)[:, :, : mel_lens.max()]
        gen_mels = mels + torch.rand_like(mels) * self.config.model.input_noise
        gen_audio, (harmonic, noise) = self.generator(gen_mels, pitches)

        # Generator
        optim_g.zero_grad(set_to_none=True)

        if self.config.model_ddsp.use_discriminator:
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

            loss_gen = gen_loss + feat_loss

            self.log(
                f"train/loss_gan",
                gen_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=pitches.shape[0],
            )

            self.log(
                f"train/loss_feat",
                feat_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=pitches.shape[0],
            )

        else:
            loss_gen = self.ddsp_loss(audio, gen_audio)

        loss_uv = self.uv_loss(gen_audio, harmonic, 1 - vuv)

        loss_gen = loss_gen + loss_uv

        self.manual_backward(loss_gen)
        optim_g.step()

        self.log(
            f"train/loss_uv",
            loss_uv,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            f"train/loss_gen",
            loss_gen,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=pitches.shape[0],
        )

        if self.config.model_ddsp.use_discriminator:
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
                batch_size=pitches.shape[0],
            )

        if self.trainer.is_last_batch:
            # Manual LR Scheduler
            if self.config.model_ddsp.use_discriminator:
                scheduler_g, scheduler_d = self.lr_schedulers()
                scheduler_g.step()
                scheduler_d.step()
            else:
                scheduler_g = self.lr_schedulers()
                scheduler_g.step()

    def validation_step(self, batch, batch_idx):
        if self.config.model_ddsp.use_discriminator:
            current_step = self.global_step // 2
        else:
            current_step = self.global_step

        pitches, audios, vuv = (
            batch["pitch"].float(),
            batch["audio"].float(),
            batch["vuv"].float(),
        )

        mel_lens = batch["audio_lens"] // self.config.hop_length

        mels = self.get_mels(audios)[:, :, : mel_lens.max()]
        gen_audio, (harmonic, noise) = self.generator(mels, pitches)
        gen_audio_mel = self.get_mels(gen_audio)[:, :, : mel_lens.max()]

        max_len = min(audios.shape[-1], gen_audio.shape[-1])

        loss_stft = self.mss_loss(audios[:, 0, :max_len], gen_audio[:, 0, :max_len])
        loss_mel = F.l1_loss(mels, gen_audio_mel)
        loss_aud = F.l1_loss(gen_audio[:, 0, :max_len], audios[:, 0, :max_len])
        loss_uv = self.uv_loss(
            gen_audio[:, :, :max_len], harmonic[:, :, :max_len], 1 - vuv
        )

        loss_valid = loss_mel + loss_aud + loss_stft + loss_uv

        self.log(
            "valid/loss_stft",
            loss_stft,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss_mel",
            loss_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss_aud",
            loss_aud,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            f"valid/loss_uv",
            loss_uv,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss",
            loss_valid,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=pitches.shape[0],
        )

        if batch_idx == 0:
            for idx, (
                mel,
                gen_mel,
                audio,
                gen_audio,
                harmonic_audio,
                noise_audio,
                mel_len,
                audio_len,
            ) in enumerate(
                zip(
                    mels.cpu().numpy(),
                    gen_audio_mel.cpu().numpy(),
                    audios.cpu().type(torch.float32).numpy(),
                    gen_audio.type(torch.float32).cpu().numpy(),
                    harmonic.type(torch.float32).cpu().numpy(),
                    noise.type(torch.float32).cpu().numpy(),
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
                    global_step=current_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gt",
                    audio[0, :audio_len],
                    global_step=current_step,
                    sample_rate=self.config.sample_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/prediction",
                    gen_audio[0, :audio_len],
                    global_step=current_step,
                    sample_rate=self.config.sample_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/harmonic",
                    harmonic_audio[0, :audio_len],
                    global_step=current_step,
                    sample_rate=self.config.sample_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/noise",
                    noise_audio[0, :audio_len],
                    global_step=current_step,
                    sample_rate=self.config.sample_rate,
                )

                plt.close(image_mels)

    def on_train_start(self) -> None:
        torch.backends.cudnn.benchmark = True

    def on_train_end(self) -> None:
        torch.backends.cudnn.benchmark = False

    def on_validation_start(self) -> None:
        torch.backends.cudnn.benchmark = False

    def on_validation_end(self) -> None:
        torch.backends.cudnn.benchmark = True
