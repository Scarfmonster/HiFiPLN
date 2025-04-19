from typing import Dict

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import (
    BatchesProcessedColumn,
    CustomBarColumn,
    CustomTimeColumn,
    ProcessingSpeedColumn,
    RichProgressBar,
    RichProgressBarTheme,
)
from rich.progress import TextColumn


class CustomSummary(RichModelSummary):
    def __init__(self):
        super().__init__(max_depth=1, header_style="bold #D4213D")


class CustomProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.theme = RichProgressBarTheme(
            description="#E9E8E7",
            progress_bar="#D4213D",
            progress_bar_finished="#D4213D",
            progress_bar_pulse="#D4213D",
            batch_progress="#E9E8E7",
            time="grey70",
            processing_speed="#E9E8E7",
            metrics="grey70",
            metrics_text_delimiter="  ",
            metrics_format=".3f",
        )

    # Only needed to make resuming work after lightning v2.5.0
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self.train_progress_bar_id is None:
            total_batches = self.total_train_batches
            train_description = self._get_train_description(trainer.current_epoch)
            self.train_progress_bar_id = self._add_task(
                total_batches, train_description
            )
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def configure_columns(self, trainer: pl.Trainer) -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                bar_width=None,
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]

    def _get_train_description(self, current_epoch: int) -> str:
        train_description = f"Epoch {current_epoch}"
        if self.trainer.max_epochs is not None and self.trainer.max_epochs > 0:
            train_description += f"/{self.trainer.max_epochs - 1}"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            train_description = (
                f"{train_description:{len(self.validation_description)}}"
            )
        return train_description

    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> Dict[str, int | str | float | Dict[str, float]]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)

        # Rename "train/loss_all" to shorter "loss"
        loss_all = items.pop("train/loss_all", None)
        if loss_all is not None:
            items["loss"] = loss_all

        # Get steps to next validation
        if hasattr(pl_module, "steps_since_last_validation"):
            ttv = trainer.val_check_interval - pl_module.steps_since_last_validation
            items["TTV"] = str(int(ttv))
        return items
