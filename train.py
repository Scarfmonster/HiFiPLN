import argparse

import lightning as pl
import soundfile
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data import VocoderDataset, collate_fn
from model.hifigan.hifigan import dynamic_range_compression
from model.hifigan.trainer import HiFiGanTrainer
from model.hifipln.trainer import HiFiPlnTrainer
from model.power.trainer import PowerTrainer
from model.utils import get_mel_transform
from model.vuv.trainer import VUVTrainer

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    pl.seed_everything(0, workers=True)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--resume", type=str, default=None)
    argparser.add_argument("--export", type=str, required=False)

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)

    if config.precision.startswith("bf16"):

        def stft(
            input: torch.Tensor,
            n_fft: int,
            hop_length: int | None = None,
            win_length: int | None = None,
            window: torch.Tensor | None = None,
            center: bool = True,
            pad_mode: str = "reflect",
            normalized: bool = False,
            onesided: bool | None = None,
            return_complex: bool | None = True,
        ) -> torch.Tensor:
            dtype = input.dtype
            input = input.float()
            if window is not None:
                window = window.float()
            return torch.functional.stft(
                input,
                n_fft,
                hop_length,
                win_length,
                window,
                center,
                pad_mode,
                normalized,
                onesided,
                return_complex,
            )

        torch.stft = stft

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=-1,
        precision=config.precision,
        val_check_interval=config.val_check,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=10,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{step}-{valid_loss:.4f}",
                save_on_train_epoch_end=False,
                save_top_k=-1,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        strategy=SingleDeviceStrategy(device=torch.device("cuda:0")),
        # detect_anomaly=True,
        logger=TensorBoardLogger("logs", name=config.type),
        benchmark=True,
        deterministic=False,
    )

    match config.type:
        case "HiFiGan":
            model = HiFiGanTrainer(config)
        case "HiFiPLN":
            model = HiFiPlnTrainer(config)
        case "VUV":
            model = VUVTrainer(config)
        case "Power":
            model = PowerTrainer(config)

    train_dataset = VocoderDataset(config, "train")
    valid_dataset = VocoderDataset(config, "valid")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.train.batch_size,
        shuffle=config.dataloader.train.shuffle,
        num_workers=config.dataloader.train.num_workers,
        pin_memory=config.dataloader.train.pin_memory,
        drop_last=config.dataloader.train.drop_last,
        persistent_workers=config.dataloader.train.persistent_workers,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.dataloader.valid.batch_size,
        shuffle=config.dataloader.valid.shuffle,
        num_workers=config.dataloader.valid.num_workers,
        pin_memory=config.dataloader.valid.pin_memory,
        drop_last=config.dataloader.valid.drop_last,
        persistent_workers=config.dataloader.valid.persistent_workers,
        collate_fn=collate_fn,
    )

    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=args.resume)
