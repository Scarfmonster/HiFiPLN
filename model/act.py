import torch
import torch.nn as nn


def swish(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(beta * x)


class Swish(nn.Module):
    def __init__(
        self,
        num_params: int,
        dim: int,
        init: int = 1,
        d2d: bool = False,
    ) -> None:
        super().__init__()
        beta = torch.zeros(num_params)
        if dim == 2:
            beta = beta[None, None, :]
        else:
            beta = beta[None, :, None]
        if d2d:
            beta = beta[:, :, :, None]

        self.beta = nn.Parameter(beta)
        self.init = init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x, self.beta + self.init)


class SiGLU(nn.Module):
    def __init__(self, num_params: int, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.act = Swish(num_params // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=self.dim)
        return x * self.act(y)


class GeGLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=self.dim)
        return x * nn.functional.gelu(y, approximate="tanh")


def snake_gamma(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    log: bool,
) -> torch.Tensor:
    alpha = alpha[None, :, None]  # line up with x to [B, C, T]
    beta = beta[None, :, None]
    gamma = gamma[None, :, None]

    if log:
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        gamma = torch.exp(gamma)
    else:
        alpha += 1.0
        beta += 1.0
        gamma += 1.0

    return x * gamma + (1.0 / (beta + 1e-8)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeGamma(nn.Module):
    def __init__(self, in_features: int, logscale: bool = False) -> None:
        super().__init__()
        self.in_features = in_features

        beta = torch.zeros(in_features, dtype=torch.float32)
        alpha = torch.zeros(in_features, dtype=torch.float32)
        gamma = torch.zeros(in_features, dtype=torch.float32)

        self.log = logscale

        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = snake_gamma(x, self.alpha, self.beta, self.gamma, self.log)

        return x


def resnake(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    log: bool,
):
    alpha = alpha[None, :, None]  # line up with x to [B, C, T]
    beta = beta[None, :, None]
    gamma = gamma[None, :, None]

    if log:
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        gamma = torch.exp(gamma)
    else:
        alpha += 1.0
        beta += 1.0
        gamma += 1.0

    g = (x * gamma).clamp(min=0.0)
    a = torch.sin(x * alpha)
    b = 1.0 / (beta + 1e-8)

    return a * b + g


class ReSnake(nn.Module):
    def __init__(
        self,
        in_features: int,
        logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.log = logscale

        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))
        self.gamma = nn.Parameter(torch.zeros(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resnake(x, self.alpha, self.beta, self.gamma, self.log)
