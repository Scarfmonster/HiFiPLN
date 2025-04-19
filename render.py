import argparse
import os
import re
from pathlib import Path

import numpy as np
import soundfile
import torch
from omegaconf import OmegaConf
from pydub import AudioSegment
from torchaudio.functional import highpass_biquad

from model.utils import STFT
from pitch import ParselmouthPE

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--shift", type=int, default=None)
    argparser.add_argument("--scale", action="store_true")
    argparser.add_argument("--gpu", action="store_true")
    argparser.add_argument("--splits", action="store_true")
    argparser.add_argument("input", type=str, nargs="+")

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)
    if config.type == "HiFiPLNv1":
        from model.hifiplnv1.generator import HiFiPLNv2

        model = HiFiPLNv2(config)
    elif config.type == "HiFiPLNv2":
        from model.hifiplnv2.generator import HiFiPLNv2

        model = HiFiPLNv2(config)
    else:
        raise ValueError(f"Unknown model type: {config.type}")

    input_file = args.model
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
                step = int(re.search(r"(?:step=)(\d+)", f).group(1))
                if step > last_epoch:
                    last_epoch = step
                    choice = i
            input_file = os.path.join(input_file, files[choice])

    print(f"Loading model from {input_file}")
    cp_dict = torch.load(input_file, map_location="cpu", weights_only=False)

    model.load_state_dict(
        {
            k.replace("generator.", ""): v
            for k, v in cp_dict["state_dict"].items()
            if k.startswith("generator.")
        },
        strict=False,
    )

    model = model.eval().remove_parametrizations()

    if args.gpu:
        model = model.cuda()

    pitch_extractor = ParselmouthPE(
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
        keep_zeros=config.preprocessing.pitch_extractor.keep_zeros,
        f0_min=config.preprocessing.f0_min,
        f0_max=config.preprocessing.f0_max,
    )

    spectogram_extractor = STFT(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.f_min,
        f_max=config.f_max,
        n_mels=config.n_mels,
    )

    with torch.inference_mode():
        for inp in args.input:
            inp = Path(inp)

            print(f"Processing {inp}")
            # audio, _ = librosa.load(inp, sr=config.sample_rate, mono=True)
            audio = (
                AudioSegment.from_file(inp)
                .set_channels(1)
                .set_frame_rate(config.sample_rate)
                .get_array_of_samples()
            )
            audio = np.array(audio, dtype=np.float32) / 32768
            print(f"Max: {np.max(audio)}")
            print(f"Audio length: {audio.shape[0] / config.sample_rate:.2f}s")
            audio = torch.from_numpy(audio).unsqueeze(0)
            audio = highpass_biquad(audio, config.sample_rate, config.f_min)

            mel = spectogram_extractor.get_mel(audio)
            pad_to = mel.shape[-1]
            f0, _, _ = pitch_extractor(audio, pad_to)

            f0 = f0[None, None, ...]

            shifts = []
            if args.shift is not None:
                shifts.append(args.shift)
            elif args.scale:
                shifts = list(range(-12, 13))
            else:
                shifts = [
                    0,
                ]

            out_dir = inp.parent / inp.stem
            out_dir.mkdir(exist_ok=True)

            if args.gpu:
                mel = mel.cuda(non_blocking=True)

            for shift in shifts:
                print(f"Rendering with pitch shift: {shift}")
                f0_scaled = f0 * (2 ** (shift / 12))

                if args.gpu:
                    f0_scaled = f0_scaled.cuda(non_blocking=True)

                gen_audio, (harm, noise) = model(mel, f0_scaled)
                gen_audio = gen_audio.squeeze().cpu().numpy()
                if args.splits:
                    if config.type == "HiFiPLNv2":
                        harm_r, harm_i = torch.chunk(harm, 2, dim=1)
                        harm = model.stft.istft(harm_r, harm_i, audio.shape[-1])
                        noise_r, noise_i = torch.chunk(noise, 2, dim=1)
                        noise = model.stft.istft(noise_r, noise_i, audio.shape[-1])
                    harm = harm.squeeze().cpu().numpy()
                    noise = noise.squeeze().cpu().numpy()

                out_file = out_dir / f"{shift}"

                output = out_file.with_suffix(f".wav")
                soundfile.write(output, gen_audio, config.sample_rate)
                if args.splits:
                    output = out_file.with_suffix(f".harm.wav")
                    soundfile.write(output, harm, config.sample_rate)
                    output = out_file.with_suffix(f".noise.wav")
                    soundfile.write(output, noise, config.sample_rate)
