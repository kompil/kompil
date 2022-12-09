import torch

from typing import Optional, List
from torch.nn.modules.module import Module


class Add(Module):

    __constants__ = ["sequences"]

    def __init__(self, sequences):
        super().__init__()
        assert sequences, "sequences has to be a non empty list"

        self.sequences = sequences

        for idx, sequence in enumerate(sequences):
            self.add_module(name=str(idx), module=sequence)

    def forward(self, x):
        res = self.sequences[0].forward(x)
        for sequence in self.sequences[1:]:
            res = res + sequence.forward(x)
        return res


class Mul(Module):

    __constants__ = ["sequences"]

    def __init__(self, sequences):
        super().__init__()
        assert sequences, "sequences has to be a non empty list"

        self.sequences = sequences

        for idx, sequence in enumerate(sequences):
            self.add_module(name=str(idx), module=sequence)

    def forward(self, x):
        res = self.sequences[0].forward(x)
        for sequence in self.sequences[1:]:
            res = res * sequence.forward(x)
        return res


class Concat(Module):

    __constants__ = ["sequences", "dim"]

    def __init__(self, sequences, dim: int):
        super().__init__()
        assert sequences, "sequences has to be a non empty list"
        assert dim >= 0, "dim has to be positive or None"

        self.sequences = sequences
        self.dim = dim

        for idx, sequence in enumerate(sequences):
            self.add_module(name=str(idx), module=sequence)

    def forward(self, x):
        # Calculate results
        seq_result = []
        for i in range(0, len(self.sequences)):
            seq_result.append(self.sequences[i](x))

        # Concat and return
        return torch.cat(seq_result, dim=self.dim)


class Sum(Module):

    __constants__ = ["dim"]

    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        assert dim >= 0, "dim has to be positive or None"

        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)


class IndexSelect(Module):

    __constants__ = ["dim", "index"]

    def __init__(self, dim: int, index: List[int]):
        super().__init__()
        assert dim >= 0, "count has to be strictely positive or None"

        self.dim = dim
        self.index = torch.IntTensor(index)

    def forward(self, x):
        self.index = self.index.to(x.device)
        return torch.index_select(x, dim=self.dim, index=self.index)
