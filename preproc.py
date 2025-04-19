import argparse
import math
import os
import sys
import traceback
from multiprocessing import Pool, RLock, current_process, freeze_support
from pathlib import Path

import numpy as np
import pyworld
import torch
from natsort import os_sorted
from omegaconf import DictConfig, OmegaConf
from pydub import AudioSegment, silence
from torchaudio.functional import highpass_biquad
from tqdm import tqdm

from pitch import BasePE
from vuv import VUVEstimator


class Preprocessor:
    def __init__(self, config: DictConfig):
        self.accumulated_audio = AudioSegment.empty()
        self.hop_length = config.hop_length
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft

    def segment(
        self,
        config: DictConfig,
        audio: np.ndarray,
        harmonic: np.ndarray | None,
        aperiodic: np.ndarray | None,
        f0: np.ndarray,
        vuv: np.ndarray,
        length: float,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # Ceil the length to the nearest multiple of the hop length
        length = length * config.sample_rate
        length = math.ceil(length / config.hop_length)

        segments = []
        for i in range(audio.shape[-1] // (length * config.hop_length)):
            start = i * length
            stop = (i + 1) * length
            a = audio[:, start * config.hop_length : stop * config.hop_length]
            h = (
                harmonic[:, start * config.hop_length : stop * config.hop_length]
                if harmonic is not None
                else None
            )
            ap = (
                aperiodic[:, start * config.hop_length : stop * config.hop_length]
                if aperiodic is not None
                else None
            )
            f = f0[start:stop]
            if vuv is not None:
                v = vuv[start:stop]
            else:
                v = None
            segments.append((a, h, ap, f, v))
        return segments

    def save_segment(
        self,
        a: np.ndarray,
        h: np.ndarray | None,
        ap: np.ndarray | None,
        f0: np.ndarray,
        vuv: np.ndarray | None,
        target: str,
    ):
        data = {
            "audio": a.squeeze(0),
            "pitch": f0,
        }
        if h is not None:
            data["harmonic"] = h.squeeze(0)
        if ap is not None:
            data["aperiodic"] = ap.squeeze(0)
        if vuv is not None:
            data["vuv"] = vuv
        np.save(target, data)

    def process_file(
        self,
        config: DictConfig,
        args: argparse.Namespace,
        file_info: tuple[Path, str, Path],
        pitch_extractor: BasePE,
        vuv_extractor: VUVEstimator,
        separate_sp_ap: bool,
    ):
        try:
            root, voice, base_prefix, f = file_info

            if f != "accumulated":
                audio_file = root / f

                if audio_file.suffix.lower() == ".mp3":
                    allaudio = AudioSegment.from_file(audio_file, format="mp3")
                else:
                    allaudio: AudioSegment = AudioSegment.from_file(
                        audio_file, format="wav"
                    )

                allaudio = (
                    allaudio.set_channels(1)
                    .set_frame_rate(config.sample_rate)
                    .set_sample_width(2)
                )

                if allaudio.duration_seconds < 5:
                    self.accumulated_audio += allaudio
                    return
            else:
                allaudio = self.accumulated_audio
                self.accumulated_audio = AudioSegment.empty()
                f = Path(f"audio-{base_prefix}.wav")
                base_prefix = "accumulated"

            trimaudio = silence.split_on_silence(
                allaudio,
                min_silence_len=args.min_silence,
                silence_thresh=args.silence_tresh,
                keep_silence=300,
                seek_step=5,
            )

            trimaudio = sum(trimaudio, start=AudioSegment.empty())
            trimaudio = np.array(trimaudio.get_array_of_samples(), dtype=np.int16)
            trimaudio = torch.from_numpy(trimaudio)
            trimaudio = trimaudio.to(torch.float64) / 32768.0
            trimaudio = trimaudio.to(torch.float32).unsqueeze(0)

            # Trim the audio to a multiple of the hop length
            trimaudio = trimaudio[
                :, : trimaudio.shape[-1] - (trimaudio.shape[-1] % config.hop_length)
            ]

            # Skip if the audio is too short
            if trimaudio.shape[-1] < args.length * config.sample_rate:
                return

            trimaudio = highpass_biquad(trimaudio, config.sample_rate, config.f_min)

            f0, _, f0_0 = pitch_extractor(trimaudio, None)
            f0 = f0.cpu().numpy()

            if config.preprocessing.vuv:
                if not config.preprocessing.vuv_from_f0:
                    vuv = vuv_extractor.get_vuv(trimaudio, f0_0)
                else:
                    vuv = vuv_extractor.from_f0(f0_0)

            if config.preprocessing.oversampling > 1:
                target_f0 = trimaudio.shape[-1] // config.hop_length
                if len(f0) > target_f0 * config.preprocessing.oversampling:
                    f0 = f0[: target_f0 * config.preprocessing.oversampling]
                elif len(f0) < target_f0 * config.preprocessing.oversampling:
                    f0 = np.pad(
                        f0,
                        (0, target_f0 * config.preprocessing.oversampling - len(f0)),
                        mode="constant",
                        constant_values=(f0[-1],),
                    )
                f0 = f0.reshape(-1, config.preprocessing.oversampling).mean(axis=1)

            if config.preprocessing.vuv:
                vuv = vuv[: f0.shape[-1]]

            harmonic, aperiodic = None, None
            if separate_sp_ap:
                harmonic, aperiodic = self.separate_audio(trimaudio.numpy(), f0)

            if voice == "NULL":
                voice = ""

            voice_dir = os.path.join(args.output, voice)
            os.makedirs(voice_dir, exist_ok=True)

            if args.length > 0.0:
                if harmonic is not None:
                    harmonic = harmonic.numpy()
                if aperiodic is not None:
                    aperiodic = aperiodic.numpy()
                segments = self.segment(
                    config,
                    trimaudio.numpy(),
                    harmonic,
                    aperiodic,
                    f0,
                    vuv,
                    args.length,
                )

                for i, (a, h, ap, f0, v) in enumerate(segments):
                    filename = f"{base_prefix}-{f.stem}-{i:03d}{f.suffix.lower()}.npy"
                    if filename.startswith("-"):
                        filename = filename[1:]
                    target = os.path.join(args.output, voice, filename)
                    self.save_segment(a, h, ap, f0, v, target)
            else:
                filename = f"{base_prefix}-{f.stem}{f.suffix.lower()}.npy"
                if filename.startswith("-"):
                    filename = filename[1:]
                target = os.path.join(args.output, voice, filename)

                self.save_segment(
                    trimaudio.numpy(),
                    harmonic,
                    aperiodic,
                    f0,
                    vuv,
                    target,
                )
        except Exception as e:
            print(f"Error processing {root/f}: {e}")
            print(traceback.format_exc())
            sys.stdout.flush()
            sys.exit(1)

    def separate_audio(self, audio, f0):
        audio = audio.astype(np.double) + np.random.randn(*audio.shape) * 1e-5
        f0 = f0.astype(np.double)
        audio = audio[0]

        wav_frames = (audio.shape[0] + self.hop_length - 1) // self.hop_length
        f0_frames = f0.shape[0]
        if f0_frames < wav_frames:
            f0 = np.pad(
                f0,
                (0, wav_frames - f0_frames),
                mode="constant",
                constant_values=(f0[0], f0[-1]),
            )
        elif f0_frames > wav_frames:
            f0 = f0[:wav_frames]

        time_step = self.hop_length / self.sample_rate
        t = np.arange(0, wav_frames) * time_step
        sp = pyworld.cheaptrick(audio, f0, t, self.sample_rate, fft_size=self.n_fft)
        ap = pyworld.d4c(audio, f0, t, self.sample_rate, fft_size=self.n_fft)

        harmonic = pyworld.synthesize(
            f0,
            np.clip(sp * (1 - ap * ap), a_min=1e-16, a_max=None),  # clip to avoid zeros
            np.zeros_like(ap),
            self.sample_rate,
            frame_period=time_step * 1000,
        )
        harmonic = torch.from_numpy(harmonic).float()

        aperiodic = pyworld.synthesize(
            f0,
            sp * ap * ap,
            np.ones_like(ap),
            self.sample_rate,
            frame_period=time_step * 1000,
        )
        aperiodic = torch.from_numpy(aperiodic).float()

        harmonic = harmonic[None, :]
        aperiodic = aperiodic[None, :]

        return harmonic, aperiodic

    def process_file_list(
        self, config: DictConfig, args: argparse.Namespace, files: np.ndarray
    ):
        current = current_process()
        pos = current._identity[0] - 1

        hop_length = config.hop_length

        if config.preprocessing.vuv:
            vuv_extractor = VUVEstimator(config)
        else:
            vuv_extractor = None

        if config.preprocessing.oversampling > 1:
            hop_length = hop_length // config.preprocessing.oversampling

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

        accumulated = 0

        for af in tqdm(
            files,
            position=pos,
            maxinterval=1.0,
            miniters=1,
            smoothing=0.1,
            dynamic_ncols=True,
        ):
            self.process_file(
                config,
                args,
                af,
                pitch_extractor,
                vuv_extractor,
                config.preprocessing.separate_sp_ap,
            )

            if self.accumulated_audio.duration_seconds > 300:
                print(
                    f"\nProcessing {self.accumulated_audio.duration_seconds} extra seconds of accumulated audio..."
                )
                self.process_file(
                    config,
                    args,
                    (af[0], af[1], f"{accumulated}", "accumulated"),
                    pitch_extractor,
                    vuv_extractor,
                    config.preprocessing.separate_sp_ap,
                )
                accumulated += 1

        if self.accumulated_audio.duration_seconds > 0:
            print(
                f"\nProcessing {self.accumulated_audio.duration_seconds} extra seconds of accumulated audio..."
            )
            self.process_file(
                config,
                args,
                (af[0], af[1], f"{accumulated}", "accumulated"),
                pitch_extractor,
                vuv_extractor,
                config.preprocessing.separate_sp_ap,
            )

        del pitch_extractor
        del vuv_extractor
        torch.cuda.empty_cache()


def run(config: DictConfig, args: argparse.Namespace, files: np.ndarray):
    p = Preprocessor(config)
    p.process_file_list(config, args, files)


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--length", type=float, default=0.0)
    parser.add_argument("-ms", "--min-silence", type=int, default=300)
    parser.add_argument("-st", "--silence-tresh", type=float, default=-40.0)
    parser.add_argument("folders", nargs="+", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    os.makedirs(args.output, exist_ok=True)

    # Clean up the output directory
    if args.clean:
        print("Cleaning *.npy files...")
        for dirpath, _, filenames in os.walk(args.output):
            for name in filenames:
                if name.endswith(".npy"):
                    os.remove(Path(dirpath, name))
            if dirpath != args.output:
                os.rmdir(dirpath)

        print("Done!")

    # Create a list of audio files with their base folders
    audio_files = dict()
    for base_dir in args.folders:
        for root, _, files in os.walk(base_dir):
            root_list = root[len(base_dir) :].split(os.sep)
            voice = "NULL"
            if len(args.folders) == 1:
                root_list = root_list[1:]
            if len(root_list) > 0:
                voice = root_list[0]
                root_list = root_list[1:]
            if audio_files.get(voice) is None:
                audio_files[voice] = []
            base_prefix = "-".join(root_list)

            for f in files:
                if f.lower().endswith((".wav", ".mp3")):
                    audio_files[voice].append((Path(root), voice, base_prefix, Path(f)))

    # Filter audio files if the voice folder already exists
    if not args.clean:
        org_len = sum([len(afs) for afs in audio_files.values()])

        for voice in list(audio_files.keys()):
            voice_dir = os.path.join(args.output, voice)
            if os.path.exists(voice_dir):
                del audio_files[voice]

        new_len = sum([len(afs) for afs in audio_files.values()])
        print(f"Filtered {org_len - new_len} already processed files from the list")

    voices = list(audio_files.keys())
    voices = os_sorted(voices)
    splits = []

    for voice in voices:
        files = audio_files[voice]
        files = os_sorted(files, key=lambda x: x[3])
        splits.append((config, args, files))

    with Pool(
        config.preprocessing.threads, initializer=tqdm.set_lock, initargs=(RLock(),)
    ) as pool:
        pool.starmap(run, splits)
