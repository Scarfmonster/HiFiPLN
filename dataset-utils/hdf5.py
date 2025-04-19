import argparse
import os

import h5py
import numpy as np
from natsort import os_sorted

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)

    args = argparser.parse_args()

    # List folders in the input directory
    folders = [
        f for f in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, f))
    ]

    audio = None
    pitch = None
    vuv = None

    count = 0

    with h5py.File(os.path.join(args.output), "w") as dataset:
        for folder in folders:
            print(f"Processing folder {folder}")

            files = [
                f
                for f in os.listdir(os.path.join(args.input, folder))
                if f.endswith(".npy")
            ]

            files = os_sorted(files)

            if audio is None or pitch is None or vuv is None:
                data = np.load(
                    os.path.join(args.input, folder, files[0]), allow_pickle=True
                ).item()

                audio_len = data["audio"].shape[0]
                pitch_len = data["pitch"].shape[0]
                vuv_len = data["vuv"].shape[0]

                audio_shape = (1, audio_len)
                if "harmonic" in data and "aperiodic" in data:
                    audio_shape = (1, 3, audio_len)
                audio_maxshape = (None,) + audio_shape[1:]

                audio = dataset.create_dataset(
                    "audio",
                    shape=audio_shape,
                    maxshape=audio_maxshape,
                    dtype=np.float32,
                    chunks=audio_shape,
                    compression="gzip",
                    compression_opts=6,
                    shuffle=True,
                )
                pitch = dataset.create_dataset(
                    "pitch",
                    shape=(1, pitch_len),
                    maxshape=(None, pitch_len),
                    dtype=np.float32,
                    chunks=(1, pitch_len),
                )
                vuv = dataset.create_dataset(
                    "vuv",
                    shape=(1, vuv_len),
                    maxshape=(None, vuv_len),
                    dtype=np.float32,
                    chunks=(1, vuv_len),
                )

            audio.resize((audio.shape[0] + len(files),) + audio_shape[1:])
            pitch.resize((pitch.shape[0] + len(files), pitch_len))
            vuv.resize((vuv.shape[0] + len(files), vuv_len))

            for file in files:
                data = np.load(
                    os.path.join(args.input, folder, file), allow_pickle=True
                ).item()
                audio_data = data["audio"]
                if "harmonic" in data and "aperiodic" in data:
                    harmonic = np.pad(
                        data["harmonic"],
                        (0, audio_data.shape[-1] - data["harmonic"].shape[-1]),
                    )
                    aperiodic = np.pad(
                        data["aperiodic"],
                        (0, audio_data.shape[-1] - data["aperiodic"].shape[-1]),
                    )
                    audio_data = np.stack([audio_data, harmonic, aperiodic], axis=0)
                audio[count] = audio_data
                pitch[count] = data["pitch"]
                vuv[count] = data["vuv"]

                count += 1

            dataset.flush()
