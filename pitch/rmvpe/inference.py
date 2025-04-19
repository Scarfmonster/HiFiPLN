import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .constants import *
from .model import E2E0
from .spec import MelSpectrogram
from .utils import interp_f0, resample_align_curve, to_local_average_f0, to_viterbi_f0


class RMVPE:
    def __init__(self, model_path, hop_length=160):
        self.resample_kernel = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = E2E0(4, 1, (2, 2)).eval().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)

    @torch.no_grad()
    def mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        mel = F.pad(
            mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="constant"
        )
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(
                hidden=hidden, N_CLASS=N_CLASS, CONST=CONST, thred=thred
            )
            f0 = f0.cpu().numpy()
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.03, use_viterbi=False):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        audio = audio.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, 16000, lowpass_filter_width=128
                )
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(
                self.device
            )
            audio_res = self.resample_kernel[key_str](audio)
        mel = self.mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred, use_viterbi=use_viterbi)
        return f0

    def get_pitch(
        self, waveform, samplerate, length, *, hop_size, speed=1, interp_uv=False
    ):
        f0 = self.infer_from_audio(waveform, sample_rate=samplerate)
        uv = f0 == 0
        f0, uv = interp_f0(f0, uv)

        hop_size = int(np.round(hop_size * speed))
        time_step = hop_size / samplerate
        f0_res = resample_align_curve(f0, 0.01, time_step, length)
        uv_res = (
            resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
        )
        if not interp_uv:
            f0_res[uv_res] = 0
        return f0_res, uv_res
