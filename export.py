import argparse
import os
import re
import shutil
from pathlib import Path

import onnxscript
import torch
from omegaconf import DictConfig, OmegaConf
from onnxscript.onnx_opset import opset17 as op
from torch.onnx._internal import jit_utils

custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)


@onnxscript.script(custom_opset)
def Sinc(X):
    piX = op.CastLike(3.141592653589793, X)
    sinc = op.Sin(piX * X) / (piX * X)
    zero = op.CastLike(0, X)
    one = op.CastLike(1, X)
    return op.Where(X == zero, one, sinc)


def custom_sinc(g: jit_utils.GraphContext, X):
    return g.onnxscript_op(Sinc, X).setType(X.type())


torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::sinc",
    symbolic_fn=custom_sinc,
    opset_version=17,
)


class ExportableHiFiPLN(torch.nn.Module):
    def __init__(self, config: DictConfig, ckpt_path, model):
        super().__init__()
        self.model = model(config)

        if ckpt_path is not None:
            cp_dict = torch.load(ckpt_path, map_location="cpu")

            if "state_dict" not in cp_dict:
                self.model.load_state_dict(cp_dict["generator"])
            else:
                self.model.load_state_dict(
                    {
                        k.replace("generator.", ""): v
                        for k, v in cp_dict["state_dict"].items()
                        if k.startswith("generator.")
                    }
                )

        self.model.eval()
        self.model.remove_parametrizations()

    def forward(self, mel: torch.FloatTensor, f0: torch.FloatTensor):
        mel = mel.transpose(-1, -2)
        f0 = f0.unsqueeze(1)
        wav, (_, _) = self.model(mel, f0)
        wav = wav.squeeze(1)
        wav = torch.clamp(wav, -1, 1)

        return wav


class ExportableDDSP(torch.nn.Module):
    def __init__(self, config: DictConfig, ckpt_path, model):
        super().__init__()
        self.model = model(config)

        if ckpt_path is not None:
            cp_dict = torch.load(ckpt_path, map_location="cpu")

            if "state_dict" not in cp_dict:
                self.model.load_state_dict(cp_dict["generator"])
            else:
                self.model.load_state_dict(
                    {
                        k.replace("generator.", ""): v
                        for k, v in cp_dict["state_dict"].items()
                        if k.startswith("generator.")
                    }
                )

        self.model.eval()
        self.model.remove_parametrizations()

    def forward(self, mel: torch.FloatTensor, f0: torch.FloatTensor):
        mel = mel.transpose(-1, -2)
        wav, (_, _) = self.model(mel, f0)
        wav = wav.squeeze(1)
        wav = torch.clamp(wav, -1, 1)

        return wav


def main(input_file, output_path, config, best=False, dynamo=False):
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Output path {output_path} already exists, deleting")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    if input_file is not None and os.path.isdir(input_file):
        dirs = [
            f
            for f in os.listdir(input_file)
            if os.path.isdir(os.path.join(input_file, f)) and f.startswith("version_")
        ]

        if len(dirs) > 0:
            last_version = 0
            for d in dirs:
                version = int(d.split("_")[1])
                if version > last_version:
                    last_version = version
            input_file = os.path.join(
                input_file, f"version_{last_version}", "checkpoints"
            )
        else:
            input_file = os.path.join(input_file, "checkpoints")

        files = [f for f in os.listdir(input_file) if f.endswith(".ckpt")]
        if len(files) > 0:
            best_epoch = 100
            last_epoch = 0
            choice = 0
            for i, f in enumerate(files):
                if best:
                    loss = float(re.search(r"(?:loss=)(\d+\.\d+)", f).group(1))
                    if loss < best_epoch:
                        best_epoch = loss
                        choice = i
                else:
                    step = int(re.search(r"(?:step=)(\d+)", f).group(1))
                    if step > last_epoch:
                        last_epoch = step
                        choice = i
            input_file = os.path.join(input_file, files[choice])

    print(f"Exporting {input_file} to {output_path}")

    # Export ONNX
    print(f"Exporting ONNX")
    match config.type:
        case "HiFiPLNv1":
            from model.hifiplnv1.generator import HiFiPLNv1

            model = ExportableHiFiPLN(config, input_file, HiFiPLNv1)
        case "DDSP":
            from model.ddsp.generator import DDSP

            model = ExportableDDSP(config, DDSP)
    print("Model loaded")

    mel = torch.randn(1, 10, 128)
    f0 = torch.randn(1, 10)

    if not dynamo:
        torch.onnx.export(
            model,
            (mel, f0),
            output_path / f"{config.type.lower()}.onnx",
            input_names=["mel", "f0"],
            output_names=["waveform"],
            opset_version=17,
            dynamic_axes={
                "mel": {1: "n_frames"},
                "f0": {1: "n_frames"},
                "waveform": {1: "wave_length"},
            },
            training=torch.onnx.TrainingMode.EVAL,
        )
    else:
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        onnx_model = torch.onnx.dynamo_export(
            model, export_options=export_options, mel=mel, f0=f0
        )
        onnx_model.save(output_path / f"{config.type.lower()}.onnx")

    print(f"ONNX exported")

    print(f"Exported to {output_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default=None)
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--dynamo", action="store_true")
    argparser.add_argument("--best", action="store_true")

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)
    main(args.model, args.output, config, best=args.best, dynamo=args.dynamo)
