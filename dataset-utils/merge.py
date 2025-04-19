from argparse import ArgumentParser
from os import walk

from natsort import os_sorted
from pydub import AudioSegment

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="Input directory")
    parser.add_argument("output", help="Output directory")

    args = parser.parse_args()

    audio_files = []
    for _, _, files in walk(args.input):
        for f in files:
            if f.lower().endswith(".wav"):
                audio_files.append(f)

    audio_files = os_sorted(audio_files)

    common_names = []
    for f in audio_files:
        name = f
        if name.endswith(".wav"):
            name = name[:-4]
        if "_" in name:
            name = name.rsplit("_", 1)[0]
        elif name[-1].isdigit():
            name = name.rstrip("0123456789")
        if name not in common_names:
            common_names.append(name)

    for name in common_names:
        print(f"Merging {name}")
        segments = []
        fullname = ""
        for f in audio_files:
            if f.startswith(name):
                if not fullname:
                    fullname = f
                segments.append(AudioSegment.from_file(f"{args.input}/{f}"))

        if len(segments) == 1:
            name = fullname
            if name.endswith(".wav"):
                name = name[:-4]
        combined = sum(segments, start=AudioSegment.empty())
        combined.export(f"{args.output}/{name}.wav", format="wav")
