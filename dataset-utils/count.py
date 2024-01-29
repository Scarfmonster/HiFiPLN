import datetime
from argparse import ArgumentParser
from os import path, walk

from pydub import AudioSegment

parser = ArgumentParser()
parser.add_argument("folders", nargs="+", type=str)
parser.add_argument("--print", type=int, default=10000)
args = parser.parse_args()
total_length = 0.0
total_files = 0

for base_dir in args.folders:
    for root, _, files in walk(base_dir):
        folder_files = 0
        folder_length = 0.0
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                total_files += 1
                folder_files += 1
                audiofile = path.join(root, f)
                if total_files % args.print == 0:
                    print(
                        f"Processing {total_files}: {datetime.timedelta(seconds=round(total_length / 1000.0))}..."
                    )
                allaudio = AudioSegment.from_file(audiofile)
                total_length += len(allaudio)
                folder_length += len(allaudio)
        tabs = "\t" * (8 - len(root) // 8)
        root = root.replace(base_dir, "")
        print(
            f"{root}{tabs}{datetime.timedelta(seconds=round(folder_length / 1000.0))}\t\t{folder_files} files"
        )


len_dataset = datetime.timedelta(seconds=round(total_length / 1000.0))
print(f"Total length: {len_dataset}")
print(f"Total files: {total_files}")
