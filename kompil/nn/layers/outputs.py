import math
import torch
from typing import Tuple
from torch import Tensor
from torch.nn.modules.module import Module


class AsymParamOutput(Module):

    __constants__ = ["__cuts"]

    def __init__(self, cuts: Tuple[int, int, int]):
        super().__init__()
        self.__cuts = cuts
        self.weight = torch.nn.Parameter(torch.Tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))

    def _forward_420(self, x: Tensor) -> Tensor:
        c0 = self.__cuts[0]
        c1 = self.__cuts[1]
        c2 = self.__cuts[2]

        y = x[..., 0:4, :, :]
        u = x[..., 4:5, :, :]
        v = x[..., 5:6, :, :]

        o = x.clone()
        o[..., 0:4, :, :] = torch.where(y < c0, y * self.weight[0][0], y * self.weight[0][1])
        o[..., 4:5, :, :] = torch.where(u < c1, u * self.weight[1][0], u * self.weight[1][1])
        o[..., 5:6, :, :] = torch.where(v < c2, v * self.weight[2][0], v * self.weight[2][1])

        return o

    def _forward_444(self, x: Tensor) -> Tensor:
        c0 = self.__cuts[0]
        c1 = self.__cuts[1]
        c2 = self.__cuts[2]

        y = x[..., 0:1, :, :]
        u = x[..., 1:2, :, :]
        v = x[..., 2:3, :, :]

        o = x.clone()
        o[..., 0:1, :, :] = torch.where(y < c0, y * self.weight[0][0], y * self.weight[0][1])
        o[..., 1:2, :, :] = torch.where(u < c1, u * self.weight[1][0], u * self.weight[1][1])
        o[..., 2:3, :, :] = torch.where(v < c2, v * self.weight[2][0], v * self.weight[2][1])

        return o

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-3] == 6:
            return self._forward_420(x)
        else:
            return self._forward_444(x)

    def extra_repr(self) -> str:
        return f"cuts={self.__cuts}"


class BuilderAsymParamOutput:
    TYPE = "out_asymp"

    @classmethod
    def elem(cls, cuts: Tuple[int, int, int]) -> dict:
        return dict(type=cls.TYPE, cuts=cuts)

    @classmethod
    def make(cls, cuts, **kwargs) -> Module:
        return AsymParamOutput(cuts)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
