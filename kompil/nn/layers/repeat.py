import torch

from torch.nn.modules.module import Module


class Repeat(Module):

    __constants__ = ["sequence", "iterations"]

    def __init__(self, sequence, iterations):
        super().__init__()
        assert sequence
        assert int(iterations)

        self.sequence = sequence
        self.iterations = iterations

    def forward(self, x):
        for _ in range(0, self.iterations):
            x = self.sequence(x)

        return x


class ConvModule(Module):
    """
    Repeat the same module alongside a defined dimension.
    """

    __constants__ = ["module", "dim"]

    def __init__(self, module: Module, dim: int):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x):
        results = []
        for i in range(x.shape[self.dim]):
            selected = torch.select(x, self.dim, i)
            result = self.module(selected).unsqueeze(self.dim)
            results.append(result)

        return torch.cat(results, dim=self.dim)
