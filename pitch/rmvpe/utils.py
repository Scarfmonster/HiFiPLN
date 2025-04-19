import librosa
import numpy as np
import torch

from .constants import *


@torch.jit.script
def to_local_average_f0(
    hidden: torch.Tensor,
    N_CLASS: int,
    CONST: float,
    center_idx: int | None = None,
    thred: float = 0.03,
):
    idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]  # [B=1, T=1, N]
    idx_cents = idx * 20 + CONST  # [B=1, N]
    if center_idx is None:
        center = torch.argmax(hidden, dim=2, keepdim=True)  # [B, T, 1]
    else:
        center = torch.tensor(center_idx, device=hidden.device).view(1, 1, 1)
    start = torch.clip(center - 4, min=0)  # [B, T, 1]
    end = torch.clip(center + 5, max=N_CLASS)  # [B, T, 1]
    idx_mask = (idx >= start) & (idx < end)  # [B, T, N]
    weights = hidden * idx_mask  # [B, T, N]
    product_sum = torch.sum(weights * idx_cents, dim=2)  # [B, T]
    weight_sum = torch.sum(weights, dim=2)  # [B, T]
    cents = product_sum / (
        weight_sum + (weight_sum == 0)
    )  # avoid dividing by zero, [B, T]
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred  # [B, T]
    f0 = f0 * ~uv
    return f0.squeeze(0)


def to_viterbi_f0(hidden, thred=0.03):
    # Create viterbi transition matrix
    if not hasattr(to_viterbi_f0, "transition"):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_f0.transition = transition

    # Convert to probability
    prob = hidden.squeeze(0).cpu().numpy()
    prob = prob.T
    prob = prob / prob.sum(axis=0)

    # Perform viterbi decoding
    path = librosa.sequence.viterbi(prob, to_viterbi_f0.transition).astype(np.int64)
    center = torch.from_numpy(path).unsqueeze(0).unsqueeze(-1).to(hidden.device)

    return to_local_average_f0(hidden, center=center, thred=thred)


def resample_align_curve(
    points: np.ndarray,
    original_timestep: float,
    target_timestep: float,
    align_length: int,
):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points,
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate(
            (curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0
        )
    return curve_interp


def norm_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)  # avoid arithmetic error
    f0[uv] = -np.inf
    return f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2**f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0
