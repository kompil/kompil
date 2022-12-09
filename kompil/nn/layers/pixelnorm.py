import torch

from torch.nn.modules.module import Module


def pixelnorm(x: torch.Tensor, eps: float = 1e-8):
    return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdims=True) + eps)


class PixelNorm(Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return pixelnorm(x, self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
