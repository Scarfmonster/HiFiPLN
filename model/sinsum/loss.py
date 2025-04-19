import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F


def envelope_loss(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
) -> torch.Tensor:
    window = 128
    stride = 32
    loss = F.l1_loss(
        F.max_pool1d(x_true, window, stride),
        F.max_pool1d(x_pred, window, stride),
    ) + F.l1_loss(
        F.max_pool1d(-x_true, window, stride),
        F.max_pool1d(-x_pred, window, stride),
    )
    return loss


def symmetry_loss(
    x: torch.Tensor,
    window: int,
    stride: int,
) -> torch.Tensor:
    loss = F.mse_loss(
        F.max_pool1d(x, window, stride),
        F.max_pool1d(-x, window, stride),
    )

    return loss


def generator_loss(
    gen_real: list[tuple[list[torch.Tensor], torch.Tensor]],
    gen_pred: list[tuple[list[torch.Tensor], torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_gen = torch.zeros(1, device=gen_real[0][1].device, dtype=gen_real[0][1].dtype)
    loss_feat = torch.zeros(1, device=gen_real[0][1].device, dtype=gen_real[0][1].dtype)
    for gr, gp in zip(gen_real, gen_pred):
        loss_gen = torch.mean((gp[1] - 1) ** 2)
        loss_feat = torch.zeros(1, device=gr[1].device, dtype=gr[1].dtype)

        for feat_real, feat_fake in zip(gr[0], gp[0]):
            loss_feat += F.l1_loss(feat_real, feat_fake)

    return loss_gen, loss_feat


def discriminator_loss(
    dis_real: list[tuple[list[torch.Tensor], torch.Tensor]],
    dis_pred: list[tuple[list[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    loss_dis = torch.zeros(1, device=dis_real[0][1].device, dtype=dis_real[0][1].dtype)

    for dr, dp in zip(dis_real, dis_pred):
        loss_dis += torch.mean((dr[1] - 1) ** 2) * 0.5 + torch.mean((dp[1]) ** 2) * 0.5

    return loss_dis


def clipping_loss(
    x: torch.Tensor,
) -> torch.Tensor:
    return F.relu(torch.abs(x) - 1.0).mean()


def amplitude_loss(
    S_true: torch.Tensor,
    noise_amplitude: torch.Tensor,
    harmonic_amplitude: torch.Tensor,
    vuv: torch.Tensor,
    hop_length: int,
) -> torch.Tensor:
    S_amp = F.max_pool1d(S_true.abs(), hop_length, hop_length).unsqueeze(1)
    S_amp = S_amp.clamp(min=0.001)

    target_noise = S_amp * (1 - vuv) + 0.001 * vuv
    loss_noise = F.l1_loss(noise_amplitude, target_noise)

    target_harmonic = harmonic_amplitude.clamp(0.0, 1.0) * vuv
    loss_harmonic = F.l1_loss(harmonic_amplitude, target_harmonic)

    loss = loss_noise + loss_harmonic

    return loss


def large_weight_loss(
    model: nn.Module,
) -> torch.Tensor:
    loss = 0.0
    count = 0.0
    for module in model.modules():
        classname = module.__class__.__name__
        if "Snake" in classname:
            params = list(module.named_parameters(recurse=False))
            for name, param in params:
                if name == "gamma":
                    loss += F.mse_loss(param, torch.zeros_like(param))
                    count += 1
        elif classname == "Swish":
            params = list(module.named_parameters(recurse=False))
            for name, param in params:
                if name == "beta":
                    loss += F.mse_loss(param, torch.zeros_like(param))
                    count += 1

    loss /= count

    return loss


def sss_loss(
    S_true_real: torch.Tensor,
    S_true_imag: torch.Tensor,
    S_pred_real: torch.Tensor,
    S_pred_imag: torch.Tensor,
) -> torch.Tensor:
    S_true_amp = torch.sqrt(S_true_real**2 + S_true_imag**2)
    S_pred_amp = torch.sqrt(S_pred_real**2 + S_pred_imag**2)
    S_true_phase = torch.atan2(S_true_imag, S_true_real)
    S_pred_phase = torch.atan2(S_pred_imag, S_pred_real)
    converge_amp = torch.mean(
        torch.linalg.norm(S_true_amp - S_pred_amp, dim=(1, 2))
        / torch.linalg.norm(S_true_amp + S_pred_amp, dim=(1, 2))
    )
    log_term = F.l1_loss(
        S_true_amp.log(),
        S_pred_amp.log(),
    )
    converge_phase = torch.mean(
        torch.linalg.norm(S_true_phase - S_pred_phase, dim=(1, 2))
        / torch.linalg.norm(S_true_phase + S_pred_phase, dim=(1, 2))
    )
    phase_term = S_true_phase - S_pred_phase
    phase_term = torch.abs(
        phase_term - torch.round(phase_term / (2 * torch.pi)) * 2 * torch.pi
    ).mean()

    loss = converge_amp + log_term + converge_phase + phase_term

    return loss


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss.
    """

    def __init__(
        self,
        n_fft: int = 111,
        overlap: int = 0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=None,
            normalized=True,
            center=False,
        )

    def forward(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
    ):
        if x_true.ndim == 3:
            x_true = x_true.squeeze(-2)
        if x_pred.ndim == 3:
            x_pred = x_pred.squeeze(-2)

        S_true = self.spec(x_true) + self.eps
        S_pred = self.spec(x_pred) + self.eps

        loss = sss_loss(
            S_true.real,
            S_true.imag,
            S_pred.real,
            S_pred.imag,
        )

        return loss


class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)

    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(
        self,
        n_ffts: list[int],
        overlap: float = 0.75,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, overlap, eps) for n_fft in n_ffts])

    def forward(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
    ):
        x_pred = x_pred[..., : x_true.shape[-1]]
        value = 0.0
        for loss in self.losses:
            value += loss(x_true, x_pred) / len(self.losses)
        return value


class RSSLoss(nn.Module):
    """Random-scale Spectral Loss."""

    def __init__(
        self,
        fft_min: int,
        fft_max: int,
        n_scale: int,
        overlap: float = 0.75,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.fft_min = fft_min
        self.fft_max = fft_max + 1
        self.n_scale = n_scale
        self.overlap = overlap
        self.eps = eps

        self.ffts = nn.ModuleDict()

        for n_fft in range(fft_min, fft_max + 1):
            self.ffts[str(n_fft)] = SSSLoss(n_fft, overlap, eps)

    def forward(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        score = 0.0

        n_ffts = torch.randint(self.fft_min, self.fft_max, (self.n_scale,))
        for n_fft in n_ffts:
            fft = self.ffts[str(int(n_fft))]
            score += fft(x_true, x_pred) / self.n_scale

        return score


def uv_loss(
    signal: torch.Tensor,
    s_h: torch.Tensor,
    uv_true: torch.Tensor,
    hop_length: float,
    eps: float = 1e-8,
    uv_tolerance: float = 0.05,
) -> torch.Tensor:
    uv_mask = F.interpolate(
        uv_true,
        scale_factor=hop_length,
        mode="linear",
        align_corners=True,
    ).squeeze(-2)[:, : signal.shape[-1]]
    signal = signal.squeeze(-2)
    s_h = s_h.squeeze(-2)
    loss = torch.mean(
        torch.linalg.norm(s_h * uv_mask, dim=1)
        / (torch.linalg.norm(signal * uv_mask, dim=1) + eps)
    )
    if loss < uv_tolerance:
        loss = loss.detach()
    return loss


class UVLoss(nn.Module):
    def __init__(self, hop_length: int, eps: float = 1e-8, uv_tolerance: float = 0.05):
        super().__init__()
        self.hop_length = float(hop_length)
        self.eps = eps
        self.uv_tolerance = uv_tolerance

    def forward(self, signal, s_h, uv_true):
        loss = uv_loss(
            signal, s_h, uv_true, self.hop_length, self.eps, self.uv_tolerance
        )
        return loss
