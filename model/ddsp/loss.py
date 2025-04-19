import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss.
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0, eps=1e-7):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=1,
            normalized=True,
            center=False,
        )

    def forward(self, x_true, x_pred):
        if x_true.ndim == 3:
            x_true = x_true.squeeze(-2)
        if x_pred.ndim == 3:
            x_pred = x_pred.squeeze(-2)

        S_true = self.spec(x_true) + self.eps
        S_pred = self.spec(x_pred) + self.eps

        converge_term = torch.mean(
            torch.linalg.norm(S_true - S_pred, dim=(1, 2))
            / torch.linalg.norm(S_true + S_pred, dim=(1, 2))
        )

        log_term = F.l1_loss(S_true.log(), S_pred.log())

        loss = converge_term + self.alpha * log_term
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

    def __init__(self, n_ffts, alpha=1.0, overlap=0.75, eps=1e-7):
        super().__init__()
        self.losses = nn.ModuleList(
            [SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts]
        )

    def forward(self, x_true, x_pred):
        x_pred = x_pred[..., : x_true.shape[-1]]
        value = 0.0
        for loss in self.losses:
            value += loss(x_true, x_pred) / len(self.losses)
        return value


class RSSLoss(nn.Module):
    """
    Random-scale Spectral Loss.
    """

    def __init__(
        self, fft_min, fft_max, n_scale, alpha=1.0, overlap=0, eps=1e-7, device="cuda"
    ):
        super().__init__()
        self.fft_min = fft_min
        self.fft_max = fft_max + 1
        self.n_scale = n_scale
        self.lossdict = {}
        for n_fft in range(fft_min, fft_max + 1):
            self.lossdict[n_fft] = SSSLoss(n_fft, alpha, overlap, eps).to(device)

    def forward(self, x_pred, x_true):
        value = 0.0
        n_ffts = torch.randint(self.fft_min, self.fft_max, (self.n_scale,))
        for n_fft in n_ffts:
            loss_func = self.lossdict[int(n_fft)]
            value += loss_func(x_true, x_pred) / self.n_scale
        return value


class UVLoss(nn.Module):
    def __init__(self, hop_length, eps=1e-8, uv_tolerance=0.05):
        super().__init__()
        self.hop_length = hop_length
        self.eps = eps
        self.uv_tolerance = uv_tolerance

    def forward(self, signal, s_h, uv_true):
        uv_mask = F.interpolate(
            uv_true,
            scale_factor=self.hop_length,
            mode="linear",
            align_corners=True,
        ).squeeze(-2)[:, : signal.shape[-1]]
        signal = signal.squeeze(-2)
        s_h = s_h.squeeze(-2)
        loss = torch.mean(
            torch.linalg.norm(s_h * uv_mask, dim=1)
            / (torch.linalg.norm(signal * uv_mask, dim=1) + self.eps)
        )
        if loss < self.uv_tolerance:
            loss = loss.detach()
        return loss
