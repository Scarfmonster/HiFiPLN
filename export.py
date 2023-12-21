import argparse
import shutil
from pathlib import Path

import onnxscript
import torch
from omegaconf import DictConfig, OmegaConf
from onnxscript.onnx_opset import opset17 as op
from torch.onnx._internal import jit_utils

from model.ddsp.generator import DDSP
from model.hifipln.generator import HiFiPLN

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
    def __init__(self, config: DictConfig, ckpt_path):
        super().__init__()
        self.model = HiFiPLN(config)

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
        wav, (harmonic, noise) = self.model(mel, f0)

        return wav


class ExportableDDSP(torch.nn.Module):
    def __init__(self, config: DictConfig, ckpt_path):
        super().__init__()
        self.model = DDSP(config)

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
        wav = self.model(mel, f0)

        return wav


def main(input_file, output_path, config):
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Output path {output_path} already exists, deleting")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {input_file} to {output_path}")

    # Export ONNX
    print(f"Exporting ONNX")
    if config.type == "HiFiPLN":
        model = ExportableHiFiPLN(config, input_file)
    else:
        model = ExportableDDSP(config, input_file)
    print("Model loaded")

    mel = torch.randn(1, 64, 128)
    f0 = torch.randn(1, 64)

    torch.onnx.export(
        model,
        (mel, f0),
        output_path / f"{config.type.lower()}.onnx",
        input_names=["mel", "f0"],
        output_names=["waveform"],
        opset_version=17,
        dynamic_axes={
            "mel": {0: "batch", 1: "n_frames"},
            "f0": {0: "batch", 1: "n_frames"},
            "waveform": {0: "batch", 2: "wave_length"},
        },
        training=torch.onnx.TrainingMode.EVAL,
    )

    print(f"ONNX exported")

    print(f"Exported to {output_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default=None)
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)
    main(args.model, args.output, config)
