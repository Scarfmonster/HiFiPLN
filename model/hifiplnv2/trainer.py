import re
import time

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo.config
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional.audio as TMF
from omegaconf import DictConfig
from torchaudio.transforms import Resample

from ..common import noise_dropout
from ..utils import STFT, AutoClip, plot_mel, plot_snakes
from .discriminator import MultiCQTDiscriminator, MultiPeriodDiscriminator
from .generator import HiFiPLNv2
from .loss import (
    MSSLoss,
    RSSLoss,
    UVLoss,
    accuracy,
    clipping_loss,
    discriminator_loss,
    envelope_loss,
    generator_loss,
    large_weight_loss,
    symmetry_loss,
)


class HiFiPlnV2Trainer(pl.LightningModule):
    def __init__(self, config: DictConfig, resume: bool = False):
        super().__init__()
        self.config = config
        self.resume = resume
        self.example_input_array = (
            torch.randn(
                config.dataloader.train.batch_size,
                config.dataset.train.segment_length // config.hop_length,
                config.n_mels,
            ),
            torch.randn(
                config.dataloader.train.batch_size,
                config.dataset.train.segment_length // config.hop_length,
            ),
        )

        self.spectogram_extractor = STFT(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )

        self.validation_mel = STFT(
            sample_rate=config.sample_rate,
            n_mels=128,
            n_fft=2048,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=config.sample_rate // 2,
        )

        # Models
        self.generator = HiFiPLNv2(config)
        self.mpd = MultiPeriodDiscriminator(config)
        self.cqtd = MultiCQTDiscriminator(config)

        # Losses
        self.mss_loss = MSSLoss(
            n_ffts=config.mss.n_ffts,
            overlap=config.mss.overlap,
            eps=config.mss.eps,
            use_mel=config.mss.use_mel,
            sample_rate=config.sample_rate,
        )
        self.mss_source_loss = RSSLoss(
            fft_min=config.mss.n_ffts[-1],
            fft_max=config.mss.n_ffts[0],
            n_scale=4,
            overlap=config.mss.overlap,
            eps=config.mss.eps,
            use_mel=False,
            sample_rate=config.sample_rate,
        )
        self.uv_loss = UVLoss(config.hop_length, uv_tolerance=config.uv_tolerance)

        self.gen_autoclip = AutoClip(
            percentile=config.optimizer.clip_percentile,
            history_size=config.optimizer.clip_history,
            max_grad=config.optimizer.clip_max,
        )
        self.mpd_autoclip = []
        for _ in range(len(self.mpd.discriminators)):
            self.mpd_autoclip.append(
                AutoClip(
                    percentile=config.optimizer.clip_percentile,
                    history_size=config.optimizer.clip_history,
                    max_grad=config.optimizer.clip_max,
                )
            )

        self.cqtd_autoclip = []
        for _ in range(len(self.cqtd.discriminators)):
            self.cqtd_autoclip.append(
                AutoClip(
                    percentile=config.optimizer.clip_percentile,
                    history_size=config.optimizer.clip_history,
                    max_grad=config.optimizer.clip_max,
                )
            )

        if config.model.compile:
            print("Enabling model compilation")

            mode = config.model.get("compile_mode", "default")
            if mode not in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]:
                raise ValueError(f"Unknown compile mode: {mode}")
            mode_no_cudagraphs = (
                "default"
                if mode in ("default", "reduce-overhead")
                else "max-autotune-no-cudagraphs"
            )
            print(f"Using compile mode: {mode} (no cudagraphs: {mode_no_cudagraphs})")

            # Compile some functions for optimization
            from .. import common

            common.noise_dropout = torch.compile(common.noise_dropout, mode=mode)

            from . import loss

            loss.accuracy = torch.compile(loss.accuracy, mode=mode_no_cudagraphs)
            loss.amplitude_loss = torch.compile(
                loss.amplitude_loss, mode=mode_no_cudagraphs
            )
            loss.clipping_loss = torch.compile(
                loss.clipping_loss, mode=mode_no_cudagraphs
            )
            loss.discriminator_loss = torch.compile(
                loss.discriminator_loss, mode=mode_no_cudagraphs
            )
            loss.envelope_loss = torch.compile(
                loss.envelope_loss, mode=mode_no_cudagraphs
            )
            loss.generator_loss = torch.compile(
                loss.generator_loss, mode=mode_no_cudagraphs
            )
            loss.mel_loss = torch.compile(loss.mel_loss, mode=mode_no_cudagraphs)
            loss.sss_loss = torch.compile(loss.sss_loss, mode=mode_no_cudagraphs)
            loss.symmetry_loss = torch.compile(
                loss.symmetry_loss, mode=mode_no_cudagraphs
            )

            self.generator.compile(mode=mode)
            self.mpd.compile(mode=mode_no_cudagraphs)
            self.cqtd.compile(mode=mode_no_cudagraphs)

            self.uv_loss.compile(mode=mode)

            self.spectogram_extractor.get_mel = torch.compile(
                self.spectogram_extractor.get_mel, mode=mode_no_cudagraphs
            )
            self.validation_mel.get_mel = torch.compile(
                self.validation_mel.get_mel, mode=mode_no_cudagraphs
            )

        else:
            # JIT some functions for optimization
            from .. import act

            act.resnake = torch.jit.script(act.resnake)
            act.snake_gamma = torch.jit.script(act.snake_gamma)
            act.swish = torch.jit.script(act.swish)

            from .. import common

            common.f0_to_phase = torch.jit.script(common.f0_to_phase)
            common.noise_dropout = torch.jit.script(common.noise_dropout)
            common.normalize = torch.jit.script(common.normalize)
            common.remove_above_fmax = torch.jit.script(common.remove_above_fmax)

            from . import loss

            loss.accuracy = torch.jit.script(loss.accuracy)
            loss.amplitude_loss = torch.jit.script(loss.amplitude_loss)
            loss.clipping_loss = torch.jit.script(loss.clipping_loss)
            loss.discriminator_loss = torch.jit.script(loss.discriminator_loss)
            loss.envelope_loss = torch.jit.script(loss.envelope_loss)
            loss.generator_loss = torch.jit.script(loss.generator_loss)
            loss.mel_loss = torch.jit.script(loss.mel_loss)
            loss.sss_loss = torch.jit.script(loss.sss_loss)
            loss.symmetry_loss = torch.jit.script(loss.symmetry_loss)
            loss.uv_loss = torch.jit.script(loss.uv_loss)

            from .. import utils

            utils.dynamic_range_compression = torch.jit.script(
                utils.dynamic_range_compression
            )

            import alias.resample

            alias.resample.upsample1d = torch.jit.script(alias.resample.upsample1d)

            import alias.filter

            alias.filter.lowpassfilter1d = torch.jit.script(
                alias.filter.lowpassfilter1d
            )

        self.automatic_optimization = False

        self.freeze_discriminator = False

        if config.get("finetune", False):
            if resume is False:
                print(f"Finetuning from {config.finetune.ckpt}")

                cp_dict = torch.load(
                    config.finetune.ckpt, map_location="cpu", weights_only=False
                )
                # Cleanup the checkpoint
                self.on_save_checkpoint(cp_dict)
                self.on_load_checkpoint(cp_dict)
                self.generator.load_state_dict(
                    {
                        k.replace("generator.", ""): v
                        for k, v in cp_dict["state_dict"].items()
                        if k.startswith("generator.")
                    },
                    strict=False,
                )
                self.mpd.load_state_dict(
                    {
                        k.replace("mpd.", ""): v
                        for k, v in cp_dict["state_dict"].items()
                        if k.startswith("mpd.")
                    }
                )
                self.cqtd.load_state_dict(
                    {
                        k.replace("cqtd.", ""): v
                        for k, v in cp_dict["state_dict"].items()
                        if k.startswith("cqtd.")
                    }
                )

            if config.finetune.freeze_generator:
                print("Freezing generator")
                for param in self.generator.layers.parameters():
                    param.requires_grad = False
                for param in self.generator.merge.parameters():
                    param.requires_grad = False
                for param in self.generator.out.parameters():
                    param.requires_grad = False
            else:
                print("Unfreezing generator")
                for param in self.generator.layers.parameters():
                    param.requires_grad = True
                for param in self.generator.merge.parameters():
                    param.requires_grad = True
                for param in self.generator.out.parameters():
                    param.requires_grad = True

            if config.finetune.freeze_source:
                print("Freezing source")
                for param in self.generator.harmonic_out.parameters():
                    param.requires_grad = False
                for param in self.generator.noise_out.parameters():
                    param.requires_grad = False
            else:
                print("Unfreezing source")
                for param in self.generator.harmonic_out.parameters():
                    param.requires_grad = True
                for param in self.generator.noise_out.parameters():
                    param.requires_grad = True

            if config.finetune.freeze_encoder:
                print("Freezing encoder")
                for param in self.generator.encoder.parameters():
                    param.requires_grad = False
                for param in self.generator.decoder_harmonic.parameters():
                    param.requires_grad = False
                for param in self.generator.decoder_noise.parameters():
                    param.requires_grad = False
            else:
                print("Unfreezing encoder")
                for param in self.generator.encoder.parameters():
                    param.requires_grad = True
                for param in self.generator.decoder_harmonic.parameters():
                    param.requires_grad = True
                for param in self.generator.decoder_noise.parameters():
                    param.requires_grad = True

            if config.finetune.freeze_mpd:
                print("Freezing MPD")
                for param in self.mpd.parameters():
                    param.requires_grad = False
            else:
                print("Unfreezing MPD")
                for param in self.mpd.parameters():
                    param.requires_grad = True

            if config.finetune.freeze_cqtd:
                print("Freezing CQTD")
                for param in self.cqtd.parameters():
                    param.requires_grad = False
            else:
                print("Unfreezing CQTD")
                for param in self.cqtd.parameters():
                    param.requires_grad = True

            if config.finetune.freeze_mpd and config.finetune.freeze_cqtd:
                self.freeze_discriminator = True

        self.resample16k = Resample(config.sample_rate, 16000)
        self.last_training_step = None
        self.steps_since_last_validation = 0

        # Discriminator output logging
        self.validation_mpd_outputs_real = []
        self.validation_mpd_outputs_fake = []
        self.validation_cqtd_outputs_real = []
        self.validation_cqtd_outputs_fake = []

        if config.norm.get("normalize", False):
            self.mel_mean = torch.tensor(config.norm.mel_mean).unsqueeze(0).unsqueeze(2)
            self.mel_std = torch.tensor(config.norm.mel_std).unsqueeze(0).unsqueeze(2)
        else:
            self.mel_mean = None
            self.mel_std = None

    def configure_adamw(
        self,
        gen_params: list[torch.nn.Parameter],
        dis_params: list[torch.nn.Parameter],
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        optim_g = torch.optim.AdamW(
            params=gen_params,
            lr=self.config.optimizer.lr,
            betas=(
                self.config.optimizer.adam_b1,
                self.config.optimizer.adam_b2,
            ),
            weight_decay=0.0,
            fused=self.config.precision == "32",
            eps=1e-8 if self.config.precision == "32" else 1e-4,
        )
        optim_d = torch.optim.AdamW(
            params=dis_params,
            lr=self.config.optimizer.lr,
            betas=(
                self.config.optimizer.adam_b1,
                self.config.optimizer.adam_b2,
            ),
            weight_decay=0.0,
            fused=self.config.precision == "32",
            eps=1e-8 if self.config.precision == "32" else 1e-4,
        )

        return (optim_g, optim_d)

    def configure_sgd(
        self,
        gen_params: list[torch.nn.Parameter],
        dis_params: list[torch.nn.Parameter],
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        optim_g = torch.optim.SGD(
            params=gen_params,
            lr=self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            nesterov=True,
            weight_decay=0.0,
            fused=self.config.precision == "32",
        )
        optim_d = torch.optim.SGD(
            params=dis_params,
            lr=self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            nesterov=True,
            weight_decay=0.0,
            fused=self.config.precision == "32",
        )

        return (optim_g, optim_d)

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        gen_params = list(self.generator.parameters())
        gen_params = filter(lambda p: p.requires_grad, gen_params)
        dis_params = list(self.mpd.parameters()) + list(self.cqtd.parameters())
        dis_params = filter(lambda p: p.requires_grad, dis_params)
        opt_type = self.config.optimizer.type.lower()
        if opt_type == "adamw":
            optim_g, optim_d = self.configure_adamw(gen_params, dis_params)
        elif opt_type == "sgd":
            optim_g, optim_d = self.configure_sgd(gen_params, dis_params)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        lr_decay = 1.0 - (
            (1.0 - self.config.optimizer.lr_decay)
            / self.config.optimizer.lr_decay_steps
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g,
            lr_decay,
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d,
            lr_decay,
        )

        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def get_input_mels(self, x: torch.Tensor) -> torch.Tensor:
        mels = self.spectogram_extractor.get_mel(x.squeeze(1))
        return mels

    def get_mels(self, x: torch.Tensor) -> torch.Tensor:
        mels = self.validation_mel.get_mel(x.squeeze(1))
        return mels

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        current_step = self.global_step // 2
        self.steps_since_last_validation += 1

        if self.config.get("finetune", False):
            if self.config.finetune.freeze_generator:
                self.generator.merge.eval()
                self.generator.layers.eval()
                self.generator.out.eval()
            if self.config.finetune.freeze_source:
                self.generator.harmonic_out.eval()
                self.generator.noise_out.eval()
            if self.config.finetune.freeze_encoder:
                self.generator.encoder.eval()
                self.generator.decoder_harmonic.eval()
                self.generator.decoder_noise.eval()
            if self.config.finetune.freeze_mpd:
                self.mpd.eval()
            if self.config.finetune.freeze_cqtd:
                self.cqtd.eval()

        optim_g, optim_d = self.optimizers()

        pitches, audio, vuv, audio_org, audio_target = (
            batch["pitch"],
            batch["audio"],
            batch["vuv"],
            batch["audio_org"],
            batch["audio_target"],
        )

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        gen_mels = self.get_input_mels(audio)[:, :, : mel_lens.max()]
        mels_target = self.get_mels(audio_target)[:, :, : mel_lens.max()]

        input_noise = self.config.get("input_noise", 0.0)
        dropout = self.config.get("dropout", 0.0)
        gen_mels = noise_dropout(
            gen_mels,
            input_noise,
            dropout,
            self.mel_mean,
            self.mel_std,
        )

        self.mark_dynamc(gen_mels, pitches)
        gen_audio: torch.Tensor
        src_harmonic: torch.Tensor | None
        src_noise: torch.Tensor | None
        (
            gen_audio,
            (src_harmonic, src_noise),
        ) = self.generator(gen_mels, pitches)
        gen_audio_mel = self.get_mels(gen_audio)[:, :, : mel_lens.max()]

        src_harmonic_real, src_harmonic_imag = torch.chunk(src_harmonic, 2, dim=1)
        src_harmonic = self.generator.stft.istft(
            src_harmonic_real, src_harmonic_imag, audio.shape[-1]
        )
        src_harmonic = src_harmonic.unsqueeze(1)

        src_noise_real, src_noise_imag = torch.chunk(src_noise, 2, dim=1)
        src_noise = self.generator.stft.istft(
            src_noise_real, src_noise_imag, audio.shape[-1]
        )
        src_noise = src_noise.unsqueeze(1)

        src_waveform = src_harmonic + src_noise

        src_waveform_mel = self.get_mels(src_waveform)[:, :, : mel_lens.max()]

        # Generator
        optim_g.zero_grad(set_to_none=True)

        gen_mpd: torch.Tensor = self.mpd(gen_audio)
        real_mpd: torch.Tensor = self.mpd(audio_target)

        gen_cqtd: torch.Tensor = self.cqtd(gen_audio)
        real_cqtd: torch.Tensor = self.cqtd(audio_target)

        # Least squares loss
        loss_mpd_gen, loss_mpd_feat = generator_loss(real_mpd, gen_mpd)
        loss_cqtd_gen, loss_cqtd_feat = generator_loss(real_cqtd, gen_cqtd)

        loss_mss: torch.Tensor = self.mss_loss(audio_target, gen_audio)
        loss_mel = F.l1_loss(mels_target, gen_audio_mel)

        loss_envelope = envelope_loss(audio_target, gen_audio)
        loss_symmetry = symmetry_loss(
            gen_audio, gen_audio.shape[-1], gen_audio.shape[-1]
        )

        loss_weight = large_weight_loss(self.generator)

        max_len = min(audio.shape[-1], gen_audio.shape[-1])

        source_uv_loss = self.uv_loss(
            src_waveform[:, :, :max_len], src_harmonic[:, :, :max_len], 1 - vuv
        )

        source_mss_loss: torch.Tensor = self.mss_source_loss(audio_target, src_waveform)
        source_mel_loss = F.l1_loss(mels_target, src_waveform_mel)
        source_clipping_loss = clipping_loss(src_waveform)

        source_detach_step = self.config.get("source_detach_step", None)
        if source_detach_step is not None and current_step > source_detach_step:
            source_mss_loss = source_mss_loss.detach()
            source_mel_loss = source_mel_loss.detach()

        loss_source = (
            source_mss_loss * self.config.loss_scale.mss_source
            + source_uv_loss * self.config.loss_scale.uv_source
            + source_mel_loss * self.config.loss_scale.mel_source
            + source_clipping_loss * self.config.loss_scale.clipping_source
        )

        loss_gen_all = (
            loss_mpd_gen * self.config.loss_scale.gen_mpd
            + loss_mpd_feat * self.config.loss_scale.feat_mpd
            + loss_cqtd_gen * self.config.loss_scale.gen_cqtd
            + loss_cqtd_feat * self.config.loss_scale.feat_cqtd
            + loss_mss * self.config.loss_scale.mss
            + loss_weight * self.config.loss_scale.weight
            + loss_source * self.config.loss_scale.source
            + loss_mel * self.config.loss_scale.mel
            + loss_envelope * self.config.loss_scale.envelope
            + loss_symmetry * self.config.loss_scale.symmetry
        )

        if not torch.isnan(loss_gen_all):
            self.manual_backward(loss_gen_all)
            gen_clip = self.gen_autoclip(self.generator)
            optim_g.step()
        else:
            print("NaN detected, skipping step")
            # Check which loss is NaN
            if torch.isnan(loss_mpd_gen):
                print("loss_mpd_gen is NaN")
            if torch.isnan(loss_mpd_feat):
                print("loss_mpd_feat is NaN")
            if torch.isnan(loss_cqtd_gen):
                print("loss_cqtd_gen is NaN")
            if torch.isnan(loss_cqtd_feat):
                print("loss_cqtd_feat is NaN")
            if torch.isnan(loss_mss):
                print("loss_mss is NaN")
            if torch.isnan(loss_weight):
                print("loss_weight is NaN")
                # Detect which weight is NaN
                for name, param in self.generator.named_parameters():
                    if torch.isnan(param).any():
                        print(f"{name} is NaN")
            if torch.isnan(loss_source):
                print("loss_source is NaN")
            if torch.isnan(loss_mel):
                print("loss_mel is NaN")
            if torch.isnan(loss_envelope):
                print("loss_envelope is NaN")
            if torch.isnan(loss_symmetry):
                print("loss_symmetry is NaN")
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_by(
                1
            )
        optim_g.zero_grad(set_to_none=True)

        loss_source = source_mss_loss + source_mel_loss + source_clipping_loss

        loss_gen_all = (
            loss_mpd_gen
            + loss_mpd_feat
            + loss_cqtd_gen
            + loss_cqtd_feat
            + loss_mss
            + loss_mel
            + loss_envelope
            + loss_symmetry
        )

        acc_mpd_real, acc_mpd_fake = accuracy(real_mpd, gen_mpd)
        acc_cqtd_real, acc_cqtd_fake = accuracy(real_cqtd, gen_cqtd)

        self.log(
            "train/acc_mpd_real",
            acc_mpd_real,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )
        self.log(
            "train/acc_mpd_fake",
            acc_mpd_fake,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )
        self.log(
            "train/acc_cqtd_real",
            acc_cqtd_real,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )
        self.log(
            "train/acc_cqtd_fake",
            acc_cqtd_fake,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_mpd_gen",
            loss_mpd_gen,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_mpd_feat",
            loss_mpd_feat,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_cqtd_gen",
            loss_cqtd_gen,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_cqtd_feat",
            loss_cqtd_feat,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_mss",
            loss_mss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_mel",
            loss_mel,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_power",
            loss_envelope,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_symmetry",
            loss_symmetry,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_weight",
            loss_weight,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_source",
            loss_source,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_all",
            loss_gen_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "stats/clip_gen",
            gen_clip,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        # Discriminator Loss
        optim_d.zero_grad(set_to_none=True)

        disc_mpd_real = self.mpd(audio_org)
        disc_mpd_fake = self.mpd(gen_audio.detach())
        disc_cqtd_real = self.cqtd(audio_org)
        disc_cqtd_fake = self.cqtd(gen_audio.detach())

        # Hinge GAN loss
        loss_mpd = discriminator_loss(disc_mpd_real, disc_mpd_fake)
        loss_cqtd = discriminator_loss(disc_cqtd_real, disc_cqtd_fake)

        loss_d = loss_mpd + loss_cqtd

        if not self.freeze_discriminator:
            self.manual_backward(loss_d)
            for i in range(len(self.mpd.discriminators)):
                clip_mpd = self.mpd_autoclip[i](self.mpd.discriminators[i])

                self.log(
                    f"stats/clip_mpd_{i}",
                    clip_mpd,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=pitches.shape[0],
                )

            for i in range(len(self.cqtd.discriminators)):
                clip_cqtd = self.cqtd_autoclip[i](self.cqtd.discriminators[i])

                self.log(
                    f"stats/clip_cqtd_{i}",
                    clip_cqtd,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=pitches.shape[0],
                )

        optim_d.step()
        optim_d.zero_grad(set_to_none=True)

        self.log(
            "train/loss_mpd_disc",
            loss_mpd,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_cqtd_disc",
            loss_cqtd,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

        step_end = time.perf_counter_ns()

        if self.last_training_step != None:
            step_time = (step_end - self.last_training_step) / 1e9
            iterations_per_second = 1 / step_time

            self.log(
                "stats/iterations_per_second",
                iterations_per_second,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=pitches.shape[0],
            )

        self.last_training_step = step_end

    def validation_step(self, batch, batch_idx):
        self.steps_since_last_validation = 0
        self.last_training_step = None
        current_step = self.global_step // 2

        self.log_histograms(batch_idx, current_step)

        pitches, audios, audio_lens = (
            batch["pitch"],
            batch["audio"],
            batch["audio_lens"],
        )

        mel_lens = audio_lens // self.config.hop_length

        gen_mels = self.get_input_mels(audios)[:, :, : mel_lens.max()]
        mels = self.get_mels(audios)[:, :, : mel_lens.max()]

        self.mark_dynamc(gen_mels, pitches)
        (
            gen_audio,
            (src_harmonic, src_noise),
        ) = self.generator(gen_mels, pitches)
        gen_audio_mel = self.get_mels(gen_audio)[:, :, : mel_lens.max()]

        src_harmonic_real, src_harmonic_imag = torch.chunk(src_harmonic, 2, dim=1)
        src_harmonic = self.generator.stft.istft(
            src_harmonic_real, src_harmonic_imag, audios.shape[-1]
        )
        src_harmonic = src_harmonic.unsqueeze(1)

        src_noise_real, src_noise_imag = torch.chunk(src_noise, 2, dim=1)
        src_noise = self.generator.stft.istft(
            src_noise_real, src_noise_imag, audios.shape[-1]
        )
        src_noise = src_noise.unsqueeze(1)

        src_waveform = src_harmonic + src_noise

        max_len = min(audios.shape[-1], gen_audio.shape[-1])

        audios = audios[:, :, :max_len]
        gen_audio = gen_audio[:, :, :max_len]
        src_waveform = src_waveform[:, :, :max_len]
        src_harmonic = src_harmonic[:, :, :max_len]
        src_noise = src_noise[:, :, :max_len]

        loss_mss = self.mss_loss(
            audios,
            gen_audio,
        )
        loss_mel = F.l1_loss(mels, gen_audio_mel)
        loss_aud = F.l1_loss(gen_audio, audios)
        loss_power = envelope_loss(audios, gen_audio)
        loss_symmetry = symmetry_loss(gen_audio, self.config.sample_rate, 32)

        loss_clip_gen = clipping_loss(gen_audio)

        loss_valid = loss_mel + loss_aud + loss_mss + loss_power

        loss_clip_src = clipping_loss(src_waveform)
        loss_source_env = envelope_loss(audios, src_waveform)
        loss_source = self.mss_source_loss(
            audios,
            src_waveform,
        )
        loss_source += loss_source_env

        self.log_metrics(current_step, audios, gen_audio, audio_lens)

        if self.config.log_accuracy:
            # Discriminator accuracy
            disc_mpd_real = self.mpd(audios)
            disc_mpd_fake = self.mpd(gen_audio)

            acc_mpd_real = 0.0
            acc_mpd_fake = 0.0

            for i in range(len(disc_mpd_real)):
                self.validation_mpd_outputs_real.append(
                    disc_mpd_real[i][1].detach().cpu().numpy().flatten()
                )
                self.validation_mpd_outputs_fake.append(
                    disc_mpd_fake[i][1].detach().cpu().numpy().flatten()
                )
                acc_mpd_real += torch.mean((disc_mpd_real[i][1] > 0.5).float())
                acc_mpd_fake += torch.mean((disc_mpd_fake[i][1] < 0.5).float())

            acc_mpd_real /= len(disc_mpd_real)
            acc_mpd_fake /= len(disc_mpd_fake)

            disc_cqtd_real = self.cqtd(audios)
            disc_cqtd_fake = self.cqtd(gen_audio)

            acc_cqtd_real = 0.0
            acc_cqtd_fake = 0.0

            for i in range(len(disc_cqtd_real)):
                self.validation_cqtd_outputs_real.append(
                    disc_cqtd_real[i][1].detach().cpu().numpy().flatten()
                )
                self.validation_cqtd_outputs_fake.append(
                    disc_cqtd_fake[i][1].detach().cpu().numpy().flatten()
                )
                acc_cqtd_real += torch.mean((disc_cqtd_real[i][1] > 0.5).float())
                acc_cqtd_fake += torch.mean((disc_cqtd_fake[i][1] < 0.5).float())

            acc_cqtd_real /= len(disc_cqtd_real)
            acc_cqtd_fake /= len(disc_cqtd_fake)

            self.log(
                "valid/disc_acc_mpd_real",
                acc_mpd_real,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=pitches.shape[0],
            )

            self.log(
                "valid/disc_acc_mpd_fake",
                acc_mpd_fake,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=pitches.shape[0],
            )

            self.log(
                "valid/disc_acc_cqtd_real",
                acc_cqtd_real,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=pitches.shape[0],
            )

            self.log(
                "valid/disc_acc_cqtd_fake",
                acc_cqtd_fake,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=pitches.shape[0],
            )

        self.log(
            "valid/loss_mss",
            loss_mss,
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
            "valid/loss_power",
            loss_power,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss_symmetry",
            loss_symmetry,
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
            "valid/loss_source",
            loss_source,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss_clip_src",
            loss_clip_src,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss_clip_gen",
            loss_clip_gen,
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
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "hp_metric",
            loss_valid,
            batch_size=pitches.shape[0],
        )

        if batch_idx == 0:
            image_snakes = plot_snakes(
                self.generator, logscale=self.config.model.act_log
            )
            if image_snakes is not None:
                self.logger.experiment.add_figure(
                    f"weights/snakes",
                    image_snakes,
                    global_step=current_step,
                )

        # Log at least 8 samples
        if batch_idx * pitches.shape[0] < 8:
            mel_lens = audio_lens // 256
            mels = self.validation_mel.get_mel(audios.squeeze(1))[
                :, :, : mel_lens.max()
            ]
            gen_audio_mel = self.validation_mel.get_mel(gen_audio.squeeze(1))[
                :, :, : mel_lens.max()
            ]
            src_mels = self.validation_mel.get_mel(src_waveform.squeeze(1))[
                :, :, : mel_lens.max()
            ]

            for idx, (
                mel,
                gen_mel,
                src_mel,
                audio,
                gen_audio,
                src_audio,
                mel_len,
                audio_len,
            ) in enumerate(
                zip(
                    mels.cpu().numpy(),
                    gen_audio_mel.cpu().numpy(),
                    src_mels.cpu().numpy(),
                    audios.cpu().type(torch.float32).numpy(),
                    F.hardtanh(gen_audio).cpu().type(torch.float32).numpy(),
                    F.hardtanh(src_waveform).cpu().type(torch.float32).numpy(),
                    mel_lens.cpu().numpy(),
                    audio_lens.cpu().numpy(),
                )
            ):
                image_mels = plot_mel(
                    [
                        gen_mel[:, :mel_len],
                        mel[:, :mel_len],
                        src_mel[:, :mel_len],
                    ],
                    [
                        "Sampled Spectrogram",
                        "Ground-Truth Spectrogram",
                        "Source Spectrogram",
                    ],
                )

                idx = batch_idx * pitches.shape[0] + idx

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
                    f"sample-{idx}/wavs/src",
                    src_audio[0, :audio_len],
                    global_step=current_step,
                    sample_rate=self.config.sample_rate,
                )

                plt.close(image_mels)

    def log_histograms(self, batch_idx, current_step):
        if current_step == 0:
            return

        if batch_idx == 0:
            biases = []
            for module in self.generator.modules():
                if hasattr(module, "bias") and module.bias is not None:
                    biases.append(module.bias.detach().cpu().numpy())

            self.logger.experiment.add_histogram(
                "weights/biases",
                np.hstack(biases),
                global_step=current_step,
            )

            # Snake parameters histograms
            alphas = []
            betas = []
            gammas = []
            for layer in self.generator.modules():
                classname = layer.__class__.__name__
                if classname in ("ReSnake", "SnakeBeta", "SnakeGamma"):
                    alphas.append(layer.alpha.detach().cpu().numpy())
                    if hasattr(layer, "beta"):
                        betas.append(layer.beta.detach().cpu().numpy())
                    if hasattr(layer, "gamma"):
                        gammas.append(layer.gamma.detach().cpu().numpy())

            if self.config.model.act_log:
                for i in range(len(alphas)):
                    alphas[i] = np.exp(alphas[i])
                for i in range(len(betas)):
                    betas[i] = np.exp(betas[i])
                for i in range(len(gammas)):
                    gammas[i] = np.exp(gammas[i])
            else:
                for i in range(len(alphas)):
                    alphas[i] += 1.0
                for i in range(len(betas)):
                    betas[i] += 1.0
                for i in range(len(gammas)):
                    gammas[i] += 1.0

            if len(alphas) > 0:
                self.logger.experiment.add_histogram(
                    "weights/snakes_alpha",
                    np.hstack(alphas),
                    global_step=current_step,
                )

            if len(betas) > 0:
                self.logger.experiment.add_histogram(
                    "weights/snakes_beta",
                    np.hstack(betas),
                    global_step=current_step,
                )

            if len(gammas) > 0:
                self.logger.experiment.add_histogram(
                    "weights/snakes_gamma",
                    np.hstack(gammas),
                    global_step=current_step,
                )

            swish = []
            for layer in self.generator.modules():
                classname = layer.__class__.__name__
                if classname == "Swish":
                    swish.append(layer.beta.detach().cpu().numpy() + layer.init)

            if len(swish) > 0:
                self.logger.experiment.add_histogram(
                    "weights/swish_beta",
                    np.hstack(swish),
                    global_step=current_step,
                )

    def log_metrics(
        self,
        current_step: int,
        org_audio: torch.Tensor,
        gen_audio: torch.Tensor,
        audio_lens: torch.Tensor,
    ):
        org_audio = org_audio.detach()
        gen_audio = gen_audio.detach()
        snr = TMF.scale_invariant_signal_noise_ratio(gen_audio, org_audio).mean()
        pesq = TMF.perceptual_evaluation_speech_quality(
            self.resample16k(gen_audio),
            self.resample16k(org_audio),
            fs=16000,
            mode="wb",
        ).mean()
        sdr = TMF.scale_invariant_signal_distortion_ratio(gen_audio, org_audio).mean()
        stoi = TMF.short_time_objective_intelligibility(
            gen_audio, org_audio, self.config.sample_rate, extended=True
        ).mean()

        self.log(
            "metrics/snr",
            snr,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=org_audio.shape[0],
        )

        self.log(
            "metrics/pesq",
            pesq,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=org_audio.shape[0],
        )

        self.log(
            "metrics/sdr",
            sdr,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=org_audio.shape[0],
        )

        self.log(
            "metrics/stoi",
            stoi,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=org_audio.shape[0],
        )

    def on_train_start(self) -> None:
        torch.backends.cudnn.benchmark = True

    def on_train_end(self) -> None:
        torch.backends.cudnn.benchmark = False

    def on_validation_start(self) -> None:
        torch.backends.cudnn.benchmark = False

    def on_validation_end(self) -> None:
        torch.backends.cudnn.benchmark = True

        # Save discriminator histograms
        if len(self.validation_mpd_outputs_real) > 0:
            self.logger.experiment.add_histogram(
                "discriminator/mpd_real",
                np.hstack(self.validation_mpd_outputs_real),
                global_step=self.global_step // 2,
            )
            self.logger.experiment.add_histogram(
                "discriminator/mpd_fake",
                np.hstack(self.validation_mpd_outputs_fake),
                global_step=self.global_step // 2,
            )

            self.validation_mpd_outputs_real = []
            self.validation_mpd_outputs_fake = []

        if len(self.validation_cqtd_outputs_real) > 0:
            self.logger.experiment.add_histogram(
                "discriminator/cqtd_real",
                np.hstack(self.validation_cqtd_outputs_real),
                global_step=self.global_step // 2,
            )
            self.logger.experiment.add_histogram(
                "discriminator/cqtd_fake",
                np.hstack(self.validation_cqtd_outputs_fake),
                global_step=self.global_step // 2,
            )

    def on_load_checkpoint(self, checkpoint):
        if not self.config.get("finetune", False) or self.resume == True:
            if "gen_autoclip" in checkpoint:
                print("Loading Generator AutoClip state")
                self.gen_autoclip.load_state_dict(checkpoint["gen_autoclip"])
                del checkpoint["gen_autoclip"]
            for i in range(len(self.mpd_autoclip)):
                if f"mpd_autoclip_{i}" in checkpoint:
                    print(f"Loading MPD AutoClip {i} state")
                    self.mpd_autoclip[i].load_state_dict(
                        checkpoint[f"mpd_autoclip_{i}"]
                    )
                    del checkpoint[f"mpd_autoclip_{i}"]
            for i in range(len(self.cqtd_autoclip)):
                if f"cqtd_autoclip_{i}" in checkpoint:
                    print(f"Loading CQTD AutoClip {i} state")
                    self.cqtd_autoclip[i].load_state_dict(
                        checkpoint[f"cqtd_autoclip_{i}"]
                    )
                    del checkpoint[f"cqtd_autoclip_{i}"]

        # Delete from checkpoint keys that do not appear in self.state_dict()
        for k in list(checkpoint["state_dict"].keys()):
            if k not in self.state_dict():
                print(f"Deleting {k} from checkpoint")
                del checkpoint["state_dict"][k]

        for k in self.state_dict():
            if k not in checkpoint["state_dict"]:
                # print(f"Adding {k} to checkpoint")
                checkpoint["state_dict"][k] = self.state_dict()[k]

    def on_save_checkpoint(self, checkpoint):
        # Add AutoClip state to the checkpoint
        checkpoint["gen_autoclip"] = self.gen_autoclip.state_dict()
        for i in range(len(self.mpd_autoclip)):
            checkpoint[f"mpd_autoclip_{i}"] = self.mpd_autoclip[i].state_dict()
        for i in range(len(self.cqtd_autoclip)):
            checkpoint[f"cqtd_autoclip_{i}"] = self.cqtd_autoclip[i].state_dict()

        # Remove unnecessary static weights from the checkpoint
        regexes = (
            r"cqtd\.discriminators\..*\.cqt\..*",
            r"mss_loss\..*",
            r"mss_source_loss\..*",
            r"rss_loss\..*",
            r"uv_loss\..*",
        )
        regexes = "(?:%s)" % "|".join(regexes)

        for k in list(checkpoint["state_dict"].keys()):
            if re.match(regexes, k):
                del checkpoint["state_dict"][k]

    def mark_dynamc(self, mel, f0):
        torch._dynamo.maybe_mark_dynamic(mel, 0)
        torch._dynamo.maybe_mark_dynamic(f0, 0)
        torch._dynamo.mark_static(mel, 1)
        torch._dynamo.mark_static(f0, 1)
        torch._dynamo.maybe_mark_dynamic(mel, 2)
        torch._dynamo.maybe_mark_dynamic(f0, 2)

    def forward(self, mel: torch.FloatTensor, f0: torch.FloatTensor):
        mel = mel.transpose(-1, -2)
        f0 = f0.unsqueeze(1)
        self.mark_dynamc(mel, f0)
        wav, (_, _) = self.generator(mel, f0)
        wav = wav.squeeze(1)
        wav = torch.clamp(wav, -1, 1)

        return wav
