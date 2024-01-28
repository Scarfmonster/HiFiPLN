import datetime
from argparse import ArgumentParser
from os import makedirs, path, sep, walk

from pydub import AudioSegment, silence

parser = ArgumentParser()
parser.add_argument("--simple", action="store_true")
parser.add_argument("--length", type=int, default=15)
parser.add_argument("--min-length", type=int, default=1)
parser.add_argument("-ms", "--min-silence", type=int, default=100)
parser.add_argument("-st", "--silence-tresh", type=float, default=-40.0)
parser.add_argument("-sr", "--sampling-rate", type=int, default=44100)
parser.add_argument("--filter", type=str, default=None)
parser.add_argument("-o", "--output", type=str, default="split")
parser.add_argument("folders", nargs="+", type=str)
args = parser.parse_args()
total_length = 0.0
total_segments = 0


def split_segments(audio, length):
    length *= 1000
    segments = []
    stop = 0
    for i in range(len(audio) // length):
        start = i * length
        stop = (i + 1) * length
        segments.append(audio[start:stop])
    if len(audio[stop:]) > 0 and len(segments) > 0:
        segments[-1] += audio[stop:]
    return segments


allow_filter = ()
if args.filter is not None:
    with open(args.filter, "r") as inf:
        lines = [line.strip() for line in inf]
        allow_filter = tuple(lines)

for base_dir in args.folders:
    for root, _, files in walk(base_dir):
        root_list = root.split(sep)
        if (
            len(root_list) >= 2
            and len(allow_filter) > 0
            and root_list[1] not in allow_filter
        ):
            continue
        if len(args.folders) == 1:
            root_list = root_list[1:]
        base_prefix = "-".join(root_list)
        for f in files:
            if f.endswith((".wav")):
                base_name = path.splitext(f)[0]
                audiofile = path.join(root, f)
                makedirs(args.output, exist_ok=True)
                print("Processing {}...".format(audiofile))
                allaudio = AudioSegment.from_file(audiofile, format="wav")

                if len(allaudio) < args.min_length * 1000:
                    continue

                trimaudio = silence.split_on_silence(
                    allaudio,
                    min_silence_len=300,
                    silence_thresh=args.silence_tresh,
                    keep_silence=300,
                    seek_step=5,
                )

                trimaudio = sum(trimaudio, start=AudioSegment.empty())

                print(f"Trimmed {(len(allaudio)-len(trimaudio))/ 1000.0}s...")

                if args.simple:
                    combined_segments = split_segments(trimaudio, args.length)

                else:
                    print("Detecting silences...")
                    segments = silence.split_on_silence(
                        trimaudio,
                        min_silence_len=args.min_silence,
                        silence_thresh=args.silence_tresh,
                        keep_silence=True,
                    )

                    print("Combining segments...")
                    combined_segments = []
                    current_segment = None
                    for segment in segments:
                        if current_segment and (
                            (len(current_segment) + len(segment) <= args.length * 1000)
                            or (len(current_segment) < args.min_length * 1000)
                        ):
                            current_segment += segment
                        else:
                            if current_segment:
                                current_length = len(current_segment)
                                if current_length < args.min_length * 1000:
                                    current_segment = None
                                    continue
                                combined_segments.append(current_segment)
                                total_length += current_length
                                current_segment = None
                            current_segment = segment

                    # Add the last segment
                    if current_segment:
                        combined_segments.append(current_segment)

                tmp_segments = []

                for segment in combined_segments:
                    if len(segment) >= args.min_length * 1000:
                        tmp_segments.append(segment)
                        total_length += len(segment)

                combined_segments = tmp_segments

                print(f"Got {len(combined_segments)} segments...")
                total_segments += len(combined_segments)

                print("Saving split files...")
                for i, segment in enumerate(combined_segments):
                    target = path.join(
                        args.output, f"{base_prefix}-{base_name}-{i:03d}.wav"
                    )
                    segment = segment.set_frame_rate(args.sampling_rate).set_channels(1)
                    segment.export(target, format="wav")

len_dataset = datetime.timedelta(seconds=round(total_length / 1000.0))
print(f"Total length: {len_dataset}")
print(f"Total segments: {total_segments}")
