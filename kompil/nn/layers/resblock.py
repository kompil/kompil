import torch

from typing import Optional
from torch.nn.modules.module import Module
from torch.nn.quantized import FloatFunctional, QFunctional


class Resblock(Module):

    __constants__ = ["_select", "_dim", "_quantized"]

    def __init__(
        self, sequence: Module, select: Optional[int] = None, dim: int = 1, quantized: bool = False
    ):
        super().__init__()
        assert sequence

        self.sequence = sequence
        self.op_add = FloatFunctional() if not quantized else QFunctional()
        self.op_cat = FloatFunctional() if not quantized else QFunctional()
        self._select = select
        self._dim = dim
        self._quantized = quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle simple resblock case
        if self._select is None:
            return self.sequence(x) + x
        # Handle partial resblock case
        a, b = torch.split(x, [self._select, x.shape[self._dim] - self._select], dim=self._dim)
        y: torch.Tensor = self.sequence(b)
        c, d = torch.split(y, [self._select, y.shape[self._dim] - self._select], dim=self._dim)
        return self.op_cat.cat([self.op_add.add(a, c), d], dim=self._dim)

    def extra_repr(self) -> str:
        output = f"select={self._select}, dim={self._dim}"
        if self._quantized:
            output += ", quantized"
        return output


class BuilderResBlock:
    TYPE = "resblock"

    @classmethod
    def elem(
        cls, sequence: list, select: Optional[int] = None, dim: int = 0, quantized: bool = False
    ) -> dict:
        return dict(type=cls.TYPE, sequence=sequence, select=select, dim=dim, quantized=quantized)

    @classmethod
    def argcompat(cls, args: dict):
        if "quantized" not in args:
            args["quantized"] = False
        return args

    @classmethod
    def make(cls, input_size, sequence, select, dim, quantized, context, **kwargs) -> Module:
        from kompil.nn.topology.builder import build_topology_from_list

        # Actual dim is including batches
        adim = dim + 1

        # Handle simple resblock case
        if select is None:
            seq, _ = build_topology_from_list(
                sequence=sequence,
                input_size=input_size,
                context=context,
            )

            return Resblock(seq, select, adim)

        # Handle partial resblock case
        seq_insize = list(input_size)
        seq_insize[dim] = seq_insize[dim] - select
        seq, _ = build_topology_from_list(
            sequence=sequence,
            input_size=tuple(seq_insize),
            context=context,
        )

        return Resblock(seq, select, adim, quantized)

    @classmethod
    def predict_size(cls, input_size, sequence, select, dim, **kwargs) -> tuple:
        from kompil.nn.topology.builder import predict_size

        # Handle simple resblock case
        if select is None:
            return predict_size(sequence=sequence, input_size=input_size)

        # Handle partial resblock case
        seq_insize = list(input_size)
        seq_insize[dim] = seq_insize[dim] - select
        return predict_size(sequence=sequence, input_size=tuple(seq_insize))
