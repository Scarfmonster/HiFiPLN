from argparse import ArgumentParser
from collections import Counter
from os import path, walk

import librosa
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("folders", nargs="+", type=str)
parser.add_argument("--print", type=int, default=1000)
args = parser.parse_args()


def distribution_to_figure(
    title, x_label, y_label, items: list, values: list, zoom=0.8
):
    fig = plt.figure(figsize=(int(len(items) * zoom), 10))
    plt.bar(x=items, height=values)
    plt.tick_params(labelsize=15)
    plt.xlim(-1, len(items))
    for a, b in zip(items, values):
        plt.text(a, b, b, ha="center", va="bottom", fontsize=13)
    plt.grid()
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    return fig


if not path.exists("pitches.txt"):
    c = Counter()
    total_files = 0

    for base_dir in args.folders:
        for root, _, files in walk(base_dir):
            for f in files:
                if f.endswith((".npy")):
                    total_files += 1
                    data = np.load(path.join(root, f), allow_pickle=True)
                    item = data.item()
                    pitches = item["pitch"]
                    pitches = np.round(pitches).astype(int)
                    c.update(pitches)
                    if total_files % args.print == 0:
                        print(f"Processed {total_files}...")

    counts = sorted(c.items())

    with open("pitches.txt", "w") as out:
        for k, v in counts:
            out.write(f"{k}\t{v}\n")

base_map = dict()
max_pitch = 0.0

with open("pitches.txt", "r") as inf:
    for line in inf:
        c, b = line.strip().split("\t")
        c, b = int(c), int(b)
        if c == 0:
            continue
        if b > max_pitch:
            max_pitch = b
        n = librosa.hz_to_note(c)
        n = librosa.note_to_midi(n)
        if n in base_map:
            base_map[n] += b
        else:
            base_map[n] = b


midis = sorted(base_map.keys())
notes = [librosa.midi_to_note(m, unicode=False) for m in range(midis[0], midis[-1] + 1)]
base_plt = distribution_to_figure(
    title="Dataset Pitch Distribution Summary",
    x_label="MIDI Key",
    y_label="% of occurrences",
    items=notes,
    values=[
        round(base_map.get(m, 0) / max_pitch, 3) for m in range(midis[0], midis[-1] + 1)
    ],
)
base_plt.savefig(fname="midi_distribution.png", bbox_inches="tight", pad_inches=0.25)

midi_map = dict()

max_pitch *= 24

for i in range(-12, 13):
    for n, b in base_map.items():
        n += i
        if n in midi_map:
            midi_map[n] += b / max_pitch
        else:
            midi_map[n] = b / max_pitch
        midi_map[n] = round(midi_map[n], 2)

del_keys = [n for n in midi_map.keys() if round(midi_map[n], 2) == 0]
for n in del_keys:
    del midi_map[n]

midis = sorted(midi_map.keys())
notes = [librosa.midi_to_note(m, unicode=False) for m in range(midis[0], midis[-1] + 1)]
midi_plt = distribution_to_figure(
    title="Scaled Pitch Distribution Summary",
    x_label="MIDI Key",
    y_label="% of occurrences",
    items=notes,
    values=[midi_map.get(m, 0) for m in range(midis[0], midis[-1] + 1)],
)
midi_plt.savefig(
    fname="midi_distribution-scaled.png", bbox_inches="tight", pad_inches=0.25
)
