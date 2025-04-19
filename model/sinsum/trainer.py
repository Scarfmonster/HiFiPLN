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
from ..utils import STFT, AutoClip, plot_mel
from .generator import SinSum
from .loss import (
    MSSLoss,
    RSSLoss,
    UVLoss,
    clipping_loss,
    envelope_loss,
    large_weight_loss,
    symmetry_loss,
)

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.compiled_autograd = True


class SinSumTrainer(pl.LightningModule):
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

        self.stft_config = config.stft_resolutions

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
        self.generator = SinSum(config)

        # Losses
        self.mss_loss = MSSLoss([2048, 1024, 512, 256])
        self.rss_loss = RSSLoss(128, 2048, n_scale=4)
        self.uv_loss = UVLoss(config.hop_length, uv_tolerance=config.uv_tolerance)

        self.gen_autoclip = AutoClip(
            percentile=config.optimizer.clip_percentile,
            history_size=config.optimizer.clip_history,
            max_grad=config.optimizer.clip_max,
        )

        if config.model.compile:
            print("Enabling model compilation")
            # This is a hack to make ResampleFrac work with torch.compile
            # Otherwise resampling produces wrong lengths during validation
            from julius.resample import ResampleFrac

            ResampleFrac.forward = torch._dynamo.disable(
                ResampleFrac.forward, recursive=False
            )

            mode = "default"
            if config.model.get("max_autotune", False):
                mode = "max-autotune"

            # Compile some functions for optimization
            from .. import common

            common.noise_dropout = torch.compile(common.noise_dropout, mode=mode)

            from . import loss

            loss.amplitude_loss = torch.compile(loss.amplitude_loss, mode=mode)
            loss.clipping_loss = torch.compile(loss.clipping_loss, mode=mode)
            loss.discriminator_loss = torch.compile(loss.discriminator_loss, mode=mode)
            loss.envelope_loss = torch.compile(loss.envelope_loss, mode=mode)
            loss.generator_loss = torch.compile(loss.generator_loss, mode=mode)
            loss.sss_loss = torch.compile(loss.sss_loss, mode=mode)
            loss.symmetry_loss = torch.compile(loss.symmetry_loss, mode=mode)

            self.generator.compile(mode=mode)
            self.uv_loss.compile(mode=mode)

            self.spectogram_extractor.get_mel = torch.compile(
                self.spectogram_extractor.get_mel, mode=mode
            )
            self.validation_mel.get_mel = torch.compile(
                self.validation_mel.get_mel, mode=mode
            )

        else:
            # JIT some functions for optimization
            from .. import act

            act.swish = torch.jit.script(act.swish)

            from .. import common

            common.f0_to_phase = torch.jit.script(common.f0_to_phase)
            common.noise_dropout = torch.jit.script(common.noise_dropout)
            common.normalize = torch.jit.script(common.normalize)
            common.remove_above_fmax = torch.jit.script(common.remove_above_fmax)

            from . import loss

            loss.amplitude_loss = torch.jit.script(loss.amplitude_loss)
            loss.clipping_loss = torch.jit.script(loss.clipping_loss)
            loss.discriminator_loss = torch.jit.script(loss.discriminator_loss)
            loss.envelope_loss = torch.jit.script(loss.envelope_loss)
            loss.generator_loss = torch.jit.script(loss.generator_loss)
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

            if config.finetune.freeze_generator:
                print("Freezing generator")
                for param in self.generator.updown_block.parameters():
                    param.requires_grad = False
            else:
                print("Unfreezing generator")
                for param in self.generator.updown_block.parameters():
                    param.requires_grad = True

            if hasattr(self.generator.source, "parameters"):
                if config.finetune.freeze_source:
                    print("Freezing source")
                    for param in self.generator.source.parameters():
                        param.requires_grad = False
                else:
                    print("Unfreezing source")
                    for param in self.generator.source.parameters():
                        param.requires_grad = True

            if config.finetune.freeze_encoder:
                print("Freezing encoder")
                for param in self.generator.encoder.parameters():
                    param.requires_grad = False
            else:
                print("Unfreezing encoder")
                for param in self.generator.encoder.parameters():
                    param.requires_grad = True

        self.resample16k = Resample(config.sample_rate, 16000)
        self.last_training_step = None

    def configure_adamw(
        self,
        gen_params: list[torch.nn.Parameter],
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

        return optim_g

    def configure_sgd(
        self,
        gen_params: list[torch.nn.Parameter],
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        optim_g = torch.optim.SGD(
            params=gen_params,
            lr=self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            nesterov=True,
            weight_decay=0.0,
            fused=self.config.precision == "32",
        )
        return optim_g

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        gen_params = list(self.generator.parameters())
        opt_type = self.config.optimizer.type.lower()
        if opt_type == "adamw":
            optim_g = self.configure_adamw(gen_params)
        elif opt_type == "sgd":
            optim_g = self.configure_sgd(gen_params)
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

        return ([optim_g], [scheduler_g])

    def get_input_mels(self, x: torch.Tensor) -> torch.Tensor:
        mels = self.spectogram_extractor.get_mel(x.squeeze(1))
        return mels

    def get_mels(self, x: torch.Tensor) -> torch.Tensor:
        mels = self.validation_mel.get_mel(x.squeeze(1))
        return mels

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        current_step = self.global_step

        if self.config.get("finetune", False):
            if self.config.finetune.freeze_generator:
                self.generator.updown_block.eval()
            if self.config.finetune.freeze_source:
                self.generator.source.eval()
            if self.config.finetune.freeze_encoder:
                self.generator.encoder.eval()
            if self.config.finetune.freeze_mpd:
                self.mpd.eval()
            if self.config.finetune.freeze_cqtd:
                self.cqtd.eval()

        optim_g = self.optimizers()

        pitches, audio, vuv = (
            batch["pitch"],
            batch["audio"],
            batch["vuv"],
        )

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        gen_mels = self.get_input_mels(audio)[:, :, : mel_lens.max()]
        mels = self.get_mels(audio)[:, :, : mel_lens.max()]

        input_noise = self.config.get("input_noise", 0.0)
        dropout = self.config.get("dropout", 0.0)
        gen_mels = noise_dropout(gen_mels, input_noise, dropout)

        self.mark_dynamc(gen_mels, pitches)
        gen_audio: torch.Tensor
        gen_harmonic: torch.Tensor
        gen_noise: torch.Tensor
        (
            gen_audio,
            (gen_harmonic, gen_noise),
        ) = self.generator(gen_mels, pitches)
        gen_audio_mel = self.get_mels(gen_audio)[:, :, : mel_lens.max()]

        # Generator
        optim_g.zero_grad(set_to_none=True)

        loss_stft: torch.Tensor = self.rss_loss(audio, gen_audio)
        loss_mel = F.l1_loss(mels, gen_audio_mel)

        loss_envelope = envelope_loss(audio, gen_audio)
        loss_symmetry = symmetry_loss(
            gen_audio, gen_audio.shape[-1], gen_audio.shape[-1]
        )

        loss_weight = large_weight_loss(self.generator)

        max_len = min(audio.shape[-1], gen_audio.shape[-1])

        uv_loss = self.uv_loss(
            gen_audio[:, :, :max_len], gen_harmonic[:, :, :max_len], 1 - vuv
        )
        clip_loss = clipping_loss(gen_audio)

        loss_gen_all = (
            loss_mel * self.config.loss_scale.mel
            + loss_stft * self.config.loss_scale.stft
            + loss_weight * self.config.loss_scale.weight
            + loss_envelope * self.config.loss_scale.envelope
            + loss_symmetry * self.config.loss_scale.symmetry
            + uv_loss * self.config.loss_scale.uv
            + clip_loss * self.config.loss_scale.clipping
        )

        self.manual_backward(loss_gen_all)
        gen_clip = self.gen_autoclip(self.generator)
        optim_g.step()
        optim_g.zero_grad(set_to_none=True)

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
            "train/loss_stft",
            loss_stft,
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
            "train/loss_envelope",
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
            "train/loss_uv",
            uv_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "train/loss_clipping",
            clip_loss,
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

        scheduler_g = self.lr_schedulers()
        scheduler_g.step()

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
        self.last_training_step = None
        current_step = self.global_step

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
            _,
        ) = self.generator(gen_mels, pitches)
        gen_audio_mel = self.get_mels(gen_audio)[:, :, : mel_lens.max()]

        max_len = min(audios.shape[-1], gen_audio.shape[-1])

        loss_stft = self.mss_loss(
            audios[:, 0, :max_len],
            gen_audio[:, 0, :max_len],
        )
        loss_mel = F.l1_loss(mels, gen_audio_mel)
        loss_aud = F.l1_loss(gen_audio[:, 0, :max_len], audios[:, 0, :max_len])
        loss_power = envelope_loss(audios[:, 0, :max_len], gen_audio[:, 0, :max_len])
        loss_symmetry = symmetry_loss(gen_audio, self.config.sample_rate, 32)
        loss_clip = clipping_loss(gen_audio)
        loss_env = envelope_loss(audios[:, 0, :max_len], gen_audio[:, 0, :max_len])

        loss_valid = (
            loss_mel
            + loss_stft
            + loss_aud
            + loss_power
            + loss_clip
            + loss_env
            + loss_symmetry
        )

        self.log_metrics(current_step, audios, gen_audio, audio_lens)

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
            "valid/loss_stft",
            loss_stft,
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
            "valid/loss_power",
            loss_power,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss_clip",
            loss_clip,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
        )

        self.log(
            "valid/loss_env",
            loss_env,
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
            "valid/loss",
            loss_valid,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=pitches.shape[0],
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
                    F.hardtanh(gen_audio).cpu().type(torch.float32).numpy(),
                    mel_lens.cpu().numpy(),
                    audio_lens.cpu().numpy(),
                )
            ):
                image_mels = plot_mel(
                    [gen_mel[:, :mel_len], mel[:, :mel_len]],
                    ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
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

    def on_load_checkpoint(self, checkpoint):
        if not self.config.get("finetune", False) or self.resume == True:
            if "gen_autoclip" in checkpoint:
                print("Loading Generator AutoClip state")
                self.gen_autoclip.load_state_dict(checkpoint["gen_autoclip"])
                del checkpoint["gen_autoclip"]

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

        # Remove unnecessary static weights from the checkpoint
        regexes = (
            r"generator\.splitbands\..*",
            r"mss_loss\..*",
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
