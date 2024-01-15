import onnxruntime as rt
import onnx
import numpy as np
from omegaconf import OmegaConf
from data import VocoderDataset
from model.utils import STFT
import sys

model = onnx.load("out/HiFiPLN/hifipln.onnx")

options = rt.SessionOptions()
options.enable_profiling = True

sess = rt.InferenceSession(
    model.SerializeToString(),
    providers=rt.get_available_providers(),
    sess_options=options,
)

config = OmegaConf.load("configs/hifipln.yaml")
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

# Process dataset using the onnx model
for i, d in enumerate(valid_dataset):
    print(f"Processing {i}")
    audio, f0 = d["audio"], d["pitch"]
    mel = stft.get_mel(audio)
    mel = mel.transpose(-1, -2)

    print(mel.shape, f0.shape)

    res = sess.run(None, {"mel": mel.numpy(), "f0": f0.numpy()})

    break

prof_file = sess.end_profiling()
print(prof_file)
