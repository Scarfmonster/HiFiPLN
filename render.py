import torch
import argparse

from omegaconf import OmegaConf
from pathlib import Path
from pitch import ParselmouthPE
import librosa
from model.utils import STFT
from model.ddsp.generator import DDSP
from torchaudio.functional import highpass_biquad
import soundfile

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--shift", type=int, default=None)
    argparser.add_argument("input", type=str, nargs="+")

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)
    model = DDSP(config)

    cp_dict = torch.load(args.model, map_location="cpu")

    model.load_state_dict(
        {
            k.replace("generator.", ""): v
            for k, v in cp_dict["state_dict"].items()
            if k.startswith("generator.")
        }
    )

    model.eval()
    model.remove_parametrizations()

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

            audio, _ = librosa.load(inp, sr=config.sample_rate, mono=True)
            audio = torch.from_numpy(audio).unsqueeze(0)
            audio = highpass_biquad(audio, config.sample_rate, config.f_min)

            mel = spectogram_extractor.get_mel(audio)
            pad_to = mel.shape[-1]
            f0, _, _ = pitch_extractor(audio, pad_to)

            f0 = f0[None, None, ...]

            if args.shift:
                f0 *= 2 ** (args.shift / 12)

            gen_audio, (harm, noise) = model(mel, f0)
            gen_audio = gen_audio.squeeze().cpu().numpy()
            harm = harm.squeeze().cpu().numpy()
            noise = noise.squeeze().cpu().numpy()

            output = inp.with_suffix(".out.wav")
            soundfile.write(output, gen_audio, config.sample_rate)
            output = inp.with_suffix(".harm.wav")
            soundfile.write(output, harm, config.sample_rate)
            output = inp.with_suffix(".noise.wav")
            soundfile.write(output, noise, config.sample_rate)
