import sys

sys.path.append("..")

from hifipln.data import VocoderDataset, collate_fn  # type: ignore
from hifipln.model.utils import STFT  # type: ignore
import argparse

from omegaconf import OmegaConf
import torch

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    args = argparser.parse_args()

    config = OmegaConf.load(args.config)

    dataset = VocoderDataset(config, "train")
    dataset.pitch_shift = None
    dataset.loudness_shift = None
    dataset.hap_shift = None
    dataset.segment_length = None
    dataset.return_vuv = False
    dataset.highpass = False

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=50,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    spectogram_extractor = STFT(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.f_min,
        f_max=config.f_max,
        n_mels=config.n_mels,
    )

    # Calculate mean and std of the dataset per mel channel

    mel_mean = torch.zeros(config.n_mels, dtype=torch.float64).cuda()
    mel_std = torch.zeros(config.n_mels, dtype=torch.float64).cuda()
    pitch_mean = torch.zeros(1, dtype=torch.float64)
    pitch_std = torch.zeros(1, dtype=torch.float64)
    n = 0

    for batch in loader:
        for s in range(-12, 7):
            # audio: torch.Tensor = batch["audio"].squeeze(1)
            pitch: torch.Tensor = batch["pitch"]
            pitch = pitch * 2 ** (s / 12)
            # audio = audio.to(device="cuda", non_blocking=True)
            pitch = pitch.to(device="cpu", dtype=torch.float64, non_blocking=True)

            # mel: torch.Tensor = spectogram_extractor.get_mel(audio)
            # mel = mel.to(device="cuda", dtype=torch.float64, non_blocking=True)
            pitch_mean += pitch.mean(dim=2).sum(dim=0)
            pitch_std += pitch.std(dim=2).sum(dim=0)
            # mel_mean += mel.mean(dim=2).sum(dim=0)
            # mel_std += mel.std(dim=2).sum(dim=0)

            n += pitch.shape[0]

            if n % 100000 == 0:
                print(f"Processed {n}/{len(dataset) * 18} samples")
                # print(f"Mel Mean: {mel_mean.sum() / (n * config.n_mels):.4f}")
                # print(f"Mel Std: {mel_std.sum() / (n * config.n_mels):.4f}")
                print(f"Pitch Mean: {(pitch_mean.cpu().numpy() / n)[0]:.4f}")
                print(f"Pitch Std: {(pitch_std.cpu().numpy() / n)[0]:.4f}")

    # mel_mean /= n
    # mel_std /= n
    pitch_mean /= n
    pitch_std /= n

    # Print the mean and std of the dataset per mel channel as yaml lists
    print("Mel Mean:")
    # Print mel mean as a comma-separated list of floats, 8 values per row
    mel_mean_print = ""
    for i, value in enumerate(mel_mean):
        mel_mean_print += f"{value.item():.4f}, "
        if i % 8 == 7:
            mel_mean_print += "\n"
    print(mel_mean_print)
    print("Mel Std:")
    # Print mel std as a comma-separated list of floats, 8 values per row
    mel_std_print = ""
    for i, value in enumerate(mel_std):
        mel_std_print += f"{value.item():.4f}, "
        if i % 8 == 7:
            mel_std_print += "\n"
    print(mel_std_print)
    print(f"Pitch Mean: {pitch_mean.item():.4f}")
    print(f"Pitch Std: {pitch_std.item():.4f}")
