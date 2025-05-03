import os
import shutil


def remove_dirs(
    curr_dir="./",
    del_dirs=[
        "__pycache__",
    ],
):
    for del_dir in del_dirs:
        if del_dir in os.listdir(curr_dir):
            p = os.path.join(curr_dir, del_dir)
            print(f"Deleting {p}")
            shutil.rmtree(p)

    for dir in os.listdir(curr_dir):
        dir = os.path.join(curr_dir, dir)
        if os.path.isdir(dir):
            remove_dirs(dir, del_dirs)


remove_dirs()
