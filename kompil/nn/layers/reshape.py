import torch

from torch.nn.modules.module import Module


class Reshape(Module):

    __constants__ = ["shape"]

    def __init__(self, shape):
        super().__init__()
        self.shape = shape if not isinstance(shape, int) else (shape,)

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

    def extra_repr(self) -> str:
        return f"shape={self.shape}"
