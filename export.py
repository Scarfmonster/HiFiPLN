import shutil
import urllib
from pathlib import Path
import argparse
from omegaconf import DictConfig, OmegaConf
import lightning as pl

import torch

from model.hifipln.generator import HiFiPLN


class ExportableHiFiPLN(torch.nn.Module):
    def __init__(self, config: DictConfig, ckpt_path):
        super().__init__()
        self.model = HiFiPLN(config)

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

    def forward(self, mel: torch.Tensor, f0: torch.Tensor):
        mel = mel.transpose(-1, -2) * 2.30259
        wav, _, _ = self.model(mel, f0)

        return wav


def main(input_file, output_path, config):
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Output path {output_path} already exists, deleting")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {input_file} to {output_path}")

    checkpoint = torch.load(input_file, map_location="cpu")
    model = checkpoint["state_dict"]

    generator_params = {
        k.replace("generator.", ""): v
        for k, v in model.items()
        if k.startswith("generator.")
    }

    pt_path = output_path / "model"
    torch.save(
        {
            "generator": generator_params,
        },
        pt_path,
    )

    print(f"Exported to {pt_path}")

    # shutil.copy(config, output_path / "config.yaml")
    # print(f"Config exported")

    # Export ONNX
    print(f"Exporting ONNX")
    model = ExportableHiFiPLN(config, input_file)
    model.eval()
    print(f"Model loaded")

    mel = torch.randn(1, 64, 128)
    f0 = torch.randn(1, 64)

    torch.onnx.export(
        model,
        (mel, f0),
        output_path / "hifipln.onnx",
        input_names=["mel", "f0"],
        output_names=["waveform"],
        opset_version=16,
        dynamic_axes={
            "mel": {0: "batch", 1: "n_frames"},
            "f0": {0: "batch", 1: "n_frames"},
            "waveform": {0: "batch", 2: "wave_length"},
        },
        # training=torch.onnx.TrainingMode.TRAINING,
    )

    print(f"ONNX exported")

    # # Export license
    # if license:
    #     print(f"Exporting license")
    #     if license.startswith("http"):
    #         urllib.request.urlretrieve(license, output_path / "LICENSE")
    #     else:
    #         shutil.copy(license, output_path / "LICENSE")
    #     print(f"License exported")

    print(f"Exported to {output_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--output", type=str, default=None)

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)
    main(args.model, args.output, config)
