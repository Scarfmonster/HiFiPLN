import argparse
import os
import re

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import OmegaConf
import time

from data import VocoderDataModule
from progress import CustomProgressBar, CustomSummary

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="CONFIG_FILE",
        help="Path to the config file.",
    )
    argparser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Path to the checkpoint to resume from. Can also be a log directory, in which case the last checkpoint of the last training run will be used.",
    )
    argparser.add_argument(
        "--epochs", type=int, default=-1, help="Maximum of epochs to train for."
    )
    argparser.add_argument(
        "--steps", type=int, default=-1, help="Maximum of steps to train for."
    )

    args = argparser.parse_args()

    if args.resume is None:
        pl.seed_everything(0, workers=True, verbose=True)
    else:
        # Set random seed
        pl.seed_everything(int(time.time()), workers=True, verbose=True)

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
        max_epochs=args.epochs,
        max_steps=args.steps,
        precision=config.precision,
        val_check_interval=config.val_check,
        check_val_every_n_epoch=None,
        # num_sanity_val_steps=10,
        callbacks=[
            ModelCheckpoint(
                filename="epoch={epoch}-step={step}-loss={valid/loss:.4}",
                save_on_train_epoch_end=False,
                save_top_k=5,
                monitor="step",
                mode="max",
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor(logging_interval="step"),
            CustomProgressBar(refresh_rate=1, leave=True),
            CustomSummary(),
        ],
        strategy=SingleDeviceStrategy(device=device),
        # detect_anomaly=True,
        logger=TensorBoardLogger("logs", name=config.type),
        # benchmark=True,
        deterministic=False,
    )
    with trainer.init_module():
        match config.type:
            case "HiFiGan":
                from model.hifigan.trainer import HiFiGanTrainer

                model = HiFiGanTrainer(config)
            case "HiFiPLNv1":
                from model.hifiplnv1.trainer import HiFiPlnTrainer

                model = HiFiPlnTrainer(config)
            case "HiFiPLNv2":
                from model.hifiplnv2.trainer import HiFiPlnV2Trainer

                model = HiFiPlnV2Trainer(config, resume is not None)
            case "SinSum":
                from model.sinsum.trainer import SinSumTrainer

                model = SinSumTrainer(config, resume is not None)
            case "DDSP":
                from model.ddsp.trainer import DDSPTrainer

                model = DDSPTrainer(config)
            case _:
                raise ValueError(f"Unknown model type: {config.type}")

    dataset = VocoderDataModule(config)

    trainer.fit(model, dataset, ckpt_path=resume)
