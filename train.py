import argparse
import os
import re

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data import VocoderDataset, collate_fn

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":
    pl.seed_everything(0, workers=True)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--resume", type=str, default=None)

    args = argparser.parse_args()

    resume = args.resume
    if resume is not None and os.path.isdir(resume):
        dirs = [
            f
            for f in os.listdir(resume)
            if os.path.isdir(os.path.join(resume, f)) and f.startswith("version_")
        ]

        if len(dirs) > 0:
            last_version = 0
            for d in dirs:
                version = int(d.split("_")[1])
                if version > last_version:
                    last_version = version
            resume = os.path.join(resume, f"version_{last_version}", "checkpoints")
        else:
            resume = os.path.join(resume, "checkpoints")

        files = [f for f in os.listdir(resume) if f.endswith(".ckpt")]
        if len(files) > 0:
            last_epoch = 0
            last_filename = ""
            for f in files:
                step = int(re.search(r"(?:step=)(\d+)", f).group(1))
                if step > last_epoch:
                    last_epoch = step
                    last_filename = f
            resume = os.path.join(resume, last_filename)

    config = OmegaConf.load(args.config)

    # Check if there are enough validation files in dataset/valid
    validation_files = len(
        [f for f in os.listdir(config.dataset.valid.path) if f.endswith(".npy")]
    )
    if validation_files < config.dataloader.valid.batch_size:
        print(
            f"Not enough validation files. Please add at least {config.dataloader.valid.batch_size} files to dataset/valid and run preprocessing."
        )
        exit(1)

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

    device = torch.device("cuda:0")
    try:
        import torch_directml  # type: ignore

        device = torch_directml.device()
        print("Using DirectML: ", device)
    except ImportError as e:
        pass

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=-1,
        precision=config.precision,
        val_check_interval=config.val_check,
        check_val_every_n_epoch=None,
        # num_sanity_val_steps=10,
        callbacks=[
            ModelCheckpoint(
                filename="epoch={epoch}-step={step}-loss={valid/loss:.4}",
                save_on_train_epoch_end=False,
                save_top_k=-1,
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        strategy=SingleDeviceStrategy(device=device),
        # detect_anomaly=True,
        logger=TensorBoardLogger("logs", name=config.type),
        # benchmark=True,
        deterministic=False,
    )

    match config.type:
        case "HiFiGan":
            from model.hifigan.trainer import HiFiGanTrainer

            model = HiFiGanTrainer(config)
        case "HiFiPLNv1":
            from model.hifiplnv1.trainer import HiFiPlnTrainer

            model = HiFiPlnTrainer(config)
        case "DDSP":
            from model.ddsp.trainer import DDSPTrainer

            model = DDSPTrainer(config)

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

    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=resume)
