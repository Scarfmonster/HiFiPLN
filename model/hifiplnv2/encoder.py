import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..act import GeGLU, SiGLU, Swish
from ..layers import TransposedLayerNorm
from ..utils import get_norm, init_weights


class Encoder(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.n_mels = config.n_mels
        self.encoder_hidden = config.model.encoder_hidden
        self.encoder_layers = config.model.encoder_layers
        self.encoder_glu = config.model.encoder_glu
        self.encoder_activation = config.model.encoder_activation
        self.encoder_dropout = config.model.encoder_dropout

        norm = get_norm(config.model.norm)

        self.mel = nn.Conv1d(self.n_mels, self.encoder_hidden, 1)
        self.mel.apply(init_weights)
        self.mel = norm(self.mel)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    self.encoder_hidden,
                    norm=config.model.norm,
                    glu=self.encoder_glu,
                    activation=self.encoder_activation,
                    dropout=self.encoder_dropout,
                )
                for _ in range(self.encoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mel(x)

        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.sample_rate = config.sample_rate
        self.hop_length = config.hop_length
        self.encoder_hidden = config.model.encoder_hidden
        self.decoder_hidden = config.model.decoder_hidden
        self.decoder_layers = config.model.decoder_layers
        self.decoder_glu = config.model.decoder_glu
        self.decoder_activation = config.model.decoder_activation
        self.decoder_dropout = config.model.decoder_dropout

        norm = get_norm(config.model.norm)

        self.register_buffer("unfold_k", torch.eye(self.hop_length)[:, None, :])

        match self.decoder_activation:
            case "Swish":
                activation = Swish(self.decoder_hidden * 2, dim=1)
            case "GELU":
                activation = nn.GELU(approximate="tanh")
            case "ReLU":
                activation = nn.LeakyReLU(0.1, inplace=True)
            case _:
                raise ValueError(f"Unsupported activation: {activation}")

        self.emb = nn.Conv1d(self.encoder_hidden, self.decoder_hidden, 1)
        self.emb.apply(init_weights)
        self.emb = norm(self.emb)

        self.phase_emb = nn.Sequential(
            nn.Conv1d(self.hop_length * 2, self.decoder_hidden * 2, 3, padding=1),
            activation,
            nn.Conv1d(self.decoder_hidden * 2, self.decoder_hidden, 3, padding=1),
        )
        self.phase_emb.apply(init_weights)
        for i in range(len(self.phase_emb)):
            if isinstance(self.phase_emb[i], nn.Conv1d):
                self.phase_emb[i] = norm(self.phase_emb[i])

        self.f0_emb = nn.Conv1d(1, self.encoder_hidden, 1)
        self.f0_emb.apply(init_weights)
        self.f0_emb = norm(self.f0_emb)

        self.layers = nn.ModuleList()
        for _ in range(self.decoder_layers):
            self.layers.append(
                EncoderLayer(
                    self.decoder_hidden,
                    norm=config.model.norm,
                    glu=self.decoder_glu,
                    activation=self.decoder_activation,
                    dropout=self.decoder_dropout,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        noise: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        # sinusoid exciter signal
        sinusoid = torch.sin(2 * torch.pi * phase).transpose(1, 2)
        sinusoid_frames = F.conv1d(sinusoid, self.unfold_k, stride=self.hop_length)

        noise_frames = F.conv1d(noise, self.unfold_k, stride=self.hop_length)

        exciter = torch.cat([sinusoid_frames, noise_frames], dim=1)

        emb = self.phase_emb(exciter)
        f0 = self.f0_emb(f0)
        x = self.emb(x)

        x = x + emb + f0

        for layer in self.layers:
            x = layer(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        norm: str,
        glu: str,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()

        norm = get_norm(norm)

        match glu:
            case "SiGLU":
                glu = SiGLU(hidden * 4, dim=1)
            case "GeGLU":
                glu = GeGLU(dim=1)
            case "GLU":
                glu = nn.GLU(dim=1)
            case _:
                raise ValueError(f"Unsupported GLU type: {glu}")

        match activation:
            case "Swish":
                activation_conv = Swish(hidden * 2, dim=1)
            case "GELU":
                activation_conv = nn.GELU(approximate="tanh")
            case "ReLU":
                activation_conv = nn.LeakyReLU(0.1, inplace=True)
            case _:
                raise ValueError(f"Unsupported activation: {activation}")

        self.conv = nn.Sequential(
            TransposedLayerNorm(hidden),
            nn.Conv1d(hidden, hidden * 4, 3, padding=1),
            glu,
            nn.Conv1d(hidden * 2, hidden * 2, 31, padding=15, groups=hidden * 2),
            activation_conv,
            nn.Dropout(dropout),
            nn.Conv1d(hidden * 2, hidden, 1),
        )
        self.conv.apply(init_weights)
        for i in range(len(self.conv)):
            if isinstance(self.conv[i], nn.Conv1d):
                self.conv[i] = norm(self.conv[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(x)

        return x
