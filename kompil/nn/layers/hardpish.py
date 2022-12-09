import torch

from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import relu


def hardpish(x, weight):
    pos = relu(x)
    neg = -relu(-x)

    factor_pos = weight[0] * pos
    depth_neg = neg * relu(neg + weight[3]) / weight[1]
    factor_neg = -relu(-weight[2] * neg)
    bias = weight[4]

    y = factor_pos + depth_neg + factor_neg + bias

    return y


class HardPish(Module):
    def __init__(self):
        super().__init__()

        self.weight = Parameter(torch.Tensor([1, 7, 0.1, 3, 0]))

    def forward(self, x: Tensor) -> Tensor:
        return hardpish(x, self.weight)


class BuilderHardPish:
    TYPE = "hardpish"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, context, **kwargs) -> Module:
        return HardPish()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
