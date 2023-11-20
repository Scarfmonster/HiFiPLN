import argparse
import os
from multiprocessing import Pool, RLock, current_process, freeze_support
from pathlib import Path
from random import shuffle

import librosa
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from pitch import BasePE
from vuv import VUVEstimator


def process(
    config: DictConfig,
    audio_path: Path,
    pitch_extractor: BasePE,
    spectogram_extractor: MelSpectrogram,
    vuv_extractor: VUVEstimator,
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

    if config.preprocessing.vuv:
        pad_to = None

    f0, _, f0_0 = pitch_extractor(audio, pad_to)
    f0 = f0.cpu().numpy()

    if config.preprocessing.vuv:
        vuv = vuv_extractor.get_vuv(audio, f0_0)
        data["vuv"] = vuv
        f0 = np.interp(
            np.linspace(np.min(f0), np.max(f0), pad_to if pad_to else len(f0) // 4),
            np.linspace(np.min(f0), np.max(f0), len(f0)),
            f0,
        )

    data["pitch"] = f0

    np.save(save_path, data)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run(config, files):
    current = current_process()
    pos = current._identity[0] - 1

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

    hop_length = config.hop_length

    if config.preprocessing.vuv:
        vuv_extractor = VUVEstimator(config)
        hop_length = hop_length // 4
    else:
        vuv_extractor = None

    pitch_extractor_cls = getattr(
        __import__("pitch", fromlist=[config.preprocessing.pitch_extractor.name]),
        config.preprocessing.pitch_extractor.name,
    )
    pitch_extractor = pitch_extractor_cls(
        sample_rate=config.sample_rate,
        hop_length=hop_length,
        keep_zeros=config.preprocessing.pitch_extractor.keep_zeros,
        f0_min=config.preprocessing.f0_min,
        f0_max=config.preprocessing.f0_max,
    )

    for af in tqdm(files, position=pos):
        process(config, af, pitch_extractor, spectogram_extractor, vuv_extractor)


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

    splits = np.array_split(np.array(audio_files), config.preprocessing.threads)
    splits = [(config, files) for files in splits]

    with Pool(8, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
        pool.starmap(run, splits)
