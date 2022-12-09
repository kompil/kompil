import torch

from torch.nn.modules.module import Module


class PermuteModule(torch.nn.Module):
    __constants__ = ["dims", "actual_dims"]

    def __init__(self, *dims):
        super().__init__()

        actual_dims = [0]
        for dim in dims:
            actual_dims.append(dim + 1)

        self.dims = dims
        self.actual_dims = actual_dims

    def forward(self, x):
        return x.permute(*self.actual_dims)

    def extra_repr(self) -> str:
        return f"dims={self.dims}"
