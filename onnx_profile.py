import argparse
import time

import onnx
import onnxruntime as rt
from omegaconf import OmegaConf

from data import VocoderDataset
from model.utils import STFT


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the onnx model",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the yaml config",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="DmlExecutionProvider",
        help="""ONNX Execution Provider. Default: DmlExecutionProvider.
        Waringing: If you get message like "Specified provider is not in available provider names" you must install onnxrutime package that includes this provider.
        - onnxruntime - cpu only
        - onnxruntime-gpu - includes gpu
        - onnxruntime-directml - includes directml
        """,
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=-1,
        help="Max iterations. Default: -1 (infinite)",
    )
    args = parser.parse_args()

    model = onnx.load(args.model)

    options = rt.SessionOptions()
    options.enable_profiling = True

    sess = rt.InferenceSession(
        model.SerializeToString(),
        providers=[args.provider],
        sess_options=options,
    )

    config = OmegaConf.load(args.config)
    valid_dataset = VocoderDataset(config, "valid")

    stft = STFT(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.f_min,
        f_max=config.f_max,
        n_mels=config.n_mels,
    )

    iters = 0
    input_time = 0

    start = time.perf_counter_ns()

    for i, d in enumerate(valid_dataset):
        iters += 1
        print(f"Processing {i}. ", end="", flush=True)
        audio, f0 = d["audio"], d["pitch"]
        input_time += audio.shape[1] / config.sample_rate

        print(
            f"Input length: {audio.shape[1] / config.sample_rate:.2f} seconds. ",
            end="",
            flush=True,
        )

        mel = stft.get_mel(audio)
        mel = mel.transpose(-1, -2)

        iter_time = time.perf_counter_ns()

        sess.run(None, {"mel": mel.numpy(), "f0": f0.numpy()})

        print(
            f"Iteration time: {(time.perf_counter_ns() - iter_time) / 1e9:.2f} seconds.",
            flush=True,
        )

        if args.max_iters > 0 and iters >= args.max_iters:
            break

    end = time.perf_counter_ns()
    total_time = (end - start) / 1e9

    print()
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time: {total_time / iters:.2f} seconds")
    print()
    print(f"Total input time: {input_time:.2f} seconds")
    print(f"Average input time: {input_time / iters:.2f} seconds")
    print(f"Input to output ratio: {total_time/input_time:.2f}")

    prof_file = sess.end_profiling()
    print(f"\nProfiling results saved to: {prof_file}")


if __name__ == "__main__":
    main()
