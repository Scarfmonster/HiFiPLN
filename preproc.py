import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from torchaudio.transforms import MelSpectrogram

from tqdm import tqdm

from pitch import BasePE
import pyworld
from multiprocessing import Pool, freeze_support, RLock
from multiprocessing import current_process
from random import shuffle


def process(
    config: DictConfig,
    audio_path: Path,
    pitch_extractor: BasePE,
    spectogram_extractor: MelSpectrogram,
):
    save_path = audio_path.with_suffix(".npy")
    if save_path.exists():
        return

    data = {"path": str(audio_path)}

    audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    data["audio"] = audio

    audio = torch.from_numpy(audio).unsqueeze(0)

    if spectogram_extractor:
        mel = spectogram_extractor(audio).squeeze()
        data["mel"] = mel.cpu().numpy()
        pad_to = mel.shape[-1]
    else:
        pad_to = None

    f0, _, f0_0 = pitch_extractor(audio, pad_to)

    data["pitch"] = f0.cpu().numpy()

    vuv = get_vuv(config, audio, f0_0)

    vuv = 1 - vuv

    vuv = np.ones_like(vuv) * (vuv > 0.01)

    if config.preprocessing.vuv:
        data["vuv"] = vuv

    np.save(save_path, data)


def get_vuv(config: DictConfig, audio, f0):
    audio = audio.cpu().numpy().astype(np.float64)[0]
    f0 = f0.cpu().numpy().astype(np.float64)
    f0_len = f0.shape[0]

    time_step = config.hop_length / config.sample_rate
    wav_frames = (audio.shape[-1] + config.hop_length - 1) // config.hop_length
    t = np.arange(0, wav_frames) * time_step

    if f0.shape[0] < wav_frames - 1:
        f0 = np.pad(
            f0,
            (0, wav_frames - f0.shape[0]),
            mode="constant",
            constant_values=(f0[0], f0[-1]),
        )
    elif f0.shape[0] > wav_frames - 1:
        f0 = f0[:wav_frames]

    ap = pyworld.d4c(
        audio, f0, t, config.sample_rate, fft_size=config.n_fft, threshold=0.8
    )

    avg = np.mean(ap[:, : ap.shape[-1] // 3], axis=-1)

    return avg.astype(np.float32)[:f0_len]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run(config, files):
    current = current_process()
    pos = current._identity[0] - 1

    pitch_extractor_cls = getattr(
        __import__("pitch", fromlist=[config.preprocessing.pitch_extractor.name]),
        config.preprocessing.pitch_extractor.name,
    )
    pitch_extractor = pitch_extractor_cls(
        sample_rate=config.sample_rate,
        keep_zeros=config.preprocessing.pitch_extractor.keep_zeros,
    )

    if config.preprocessing.spectogram:
        spectogram_extractor = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )
    else:
        spectogram_extractor = None

    for af in tqdm(files, position=pos):
        process(config, af, pitch_extractor, spectogram_extractor)


if __name__ == "__main__":
    freeze_support()

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--clean", action="store_true")

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)

    if args.clean:
        print("Cleaning *.npy files...")
        for dirpath, _, dirnames in os.walk(args.path):
            for name in dirnames:
                if name.endswith(".npy"):
                    os.remove(Path(dirpath, name))

        print("Done!")

    audio_files = []

    for dirpath, _, dirnames in os.walk(args.path):
        for name in dirnames:
            if name.endswith(".wav"):
                audio_files.append(Path(dirpath, name))

    shuffle(audio_files)

    splits = np.array_split(np.array(audio_files), 8)
    splits = [(config, files) for files in splits]

    with Pool(8, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
        pool.starmap(run, splits)
