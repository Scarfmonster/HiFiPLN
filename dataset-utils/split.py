from argparse import ArgumentParser
from os import makedirs, path, sep, walk

from pydub import AudioSegment, silence
from multiprocessing import Pool, RLock, current_process, freeze_support
from tqdm import tqdm
from random import shuffle
import numpy as np


def split_segments(audio, length):
    segments = []
    stop = 0
    for i in range(len(audio) // length):
        start = i * length
        stop = (i + 1) * length
        segments.append(audio[start:stop])
    if len(audio[stop:]) > 0 and len(segments) > 0:
        segments[-1] += audio[stop:]
    return segments


def process_file(args, files):
    pos = current_process()._identity[0] - 1
    for root, base_prefix, f in tqdm(files, position=pos, desc=f"Split #{pos}"):
        base_name = path.splitext(f)[0]
        audiofile = path.join(root, f)

        target_length = round(args.length * 1000)

        if path.exists(path.join(args.output, f"{base_prefix}-{base_name}-000.wav")):
            continue

        _, extension = path.splitext(f.lower())
        if extension == ".mp3":
            allaudio = AudioSegment.from_file(audiofile, format="mp3")
        else:
            allaudio = AudioSegment.from_file(audiofile, format="wav")

        if len(allaudio) < target_length:
            continue

        trimaudio = silence.split_on_silence(
            allaudio,
            min_silence_len=args.min_silence,
            silence_thresh=args.silence_tresh,
            keep_silence=300,
            seek_step=5,
        )

        trimaudio = sum(trimaudio, start=AudioSegment.empty())

        combined_segments = split_segments(trimaudio, target_length)

        for i, segment in enumerate(combined_segments):
            filename = f"{base_prefix}-{base_name}-{i:03d}.wav"
            if filename.startswith("-"):
                filename = filename[1:]
            target = path.join(args.output, filename)
            segment = segment.set_frame_rate(args.sampling_rate).set_channels(1)
            segment.export(target, format="wav")


if __name__ == "__main__":
    freeze_support()

    parser = ArgumentParser()
    parser.add_argument("--length", type=float, default=15.0)
    parser.add_argument("-ms", "--min-silence", type=int, default=300)
    parser.add_argument("-st", "--silence-tresh", type=float, default=-40.0)
    parser.add_argument("-sr", "--sampling-rate", type=int, default=44100)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default="split")
    parser.add_argument("-t", "--threads", type=int, default=8)
    parser.add_argument("folders", nargs="+", type=str)
    args = parser.parse_args()

    allow_filter = ()
    if args.filter is not None:
        with open(args.filter, "r") as inf:
            lines = [line.strip() for line in inf]
            allow_filter = tuple(lines)

    for base_dir in args.folders:
        audio_files = []
        for root, _, files in walk(base_dir):
            root_list = root[len(base_dir) :].split(sep)
            if (
                len(root_list) >= 2
                and len(allow_filter) > 0
                and root_list[1] not in allow_filter
            ):
                continue
            if len(args.folders) == 1:
                root_list = root_list[1:]
            base_prefix = "-".join(root_list)
            makedirs(args.output, exist_ok=True)

            for f in files:
                if f.lower().endswith((".wav", ".mp3")):
                    audio_files.append((root, base_prefix, f))

        shuffle(audio_files)

        splits = np.array_split(np.array(audio_files), args.threads)
        splits = [(args, files) for files in splits]

        with Pool(args.threads, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
            pool.starmap(process_file, splits)
