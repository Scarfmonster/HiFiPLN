import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch.nn.utils.parametrizations import weight_norm

from ..utils import init_weights


class PreEncoder(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.n_mels = config.n_mels
        self.upsample_initial = config.model.upsample_initial

        self.convs1 = nn.Sequential(
            weight_norm(nn.Conv1d(self.n_mels, 256, 3, padding=1)),
            nn.GroupNorm(4, 256),
            nn.GELU(),
            weight_norm(nn.Conv1d(256, 256, 3, padding=1)),
        )

        self.norm1 = nn.LayerNorm(256)
        self.attention1 = Attention(256, heads=8, dim_head=32)

        self.convs2 = nn.Sequential(
            weight_norm(nn.Conv1d(256 + 1, 512, 3, padding=1)),
            nn.GLU(1),
            weight_norm(nn.Conv1d(256, 512, 3, padding=1)),
            nn.GLU(1),
            weight_norm(nn.Conv1d(256, 256, 3, padding=1)),
        )

        self.norm2 = nn.LayerNorm(256)
        self.attention2 = Attention(256, heads=8, dim_head=32)

        self.norm3 = nn.LayerNorm(256)
        self.post = weight_norm(nn.Linear(256, self.upsample_initial))

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x, f0):
        x = self.convs1(x)

        x = x.transpose(1, 2)
        a = self.attention1(self.norm1(x))
        x = x + a
        x = x.transpose(1, 2)

        x = torch.cat([x, f0], dim=1)
        x = self.convs2(x)

        x = x.transpose(1, 2)
        a = self.attention2(self.norm2(x))
        x = x + a
        x = self.norm3(x)
        x = self.post(x)
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, conditiondim=None):
        super().__init__()
        if conditiondim is None:
            conditiondim = dim

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(conditiondim, hidden_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(
                hidden_dim,
                dim,
            ),
        )

    def forward(self, q, kv=None, mask=None):
        # b, c, h, w = x.shape
        if kv is None:
            kv = q
        # q, kv = map(
        #     lambda t: rearrange(t, "b c t -> b t c", ), (q, kv)
        # )

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=2)

        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.heads), (q, k, v)
        )

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        with torch.backends.cuda.sdp_kernel():  # enable_math=False
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = rearrange(
            out,
            "b h t c -> b t (h c) ",
            h=self.heads,
        )
        return self.to_out(out)
