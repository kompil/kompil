import torch

from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import relu


def xelu(x, weight):
    neg = -relu(-x) * weight[1]
    pos = relu(x) * weight[2]

    far_neg = torch.where(neg > -1, neg, neg * weight[0])
    far_pos = torch.where(pos < 1, pos, pos * weight[3])

    y = far_pos + far_neg + weight[-1]

    return y


class XeLU(Module):
    def __init__(self):
        super().__init__()

        self.weight = Parameter(torch.Tensor([0.8, 0.3, 1, 0.8, 0]))

    def forward(self, x: Tensor) -> Tensor:
        return xelu(x, self.weight)


class BuilderXeLU:
    TYPE = "xelu"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, context, **kwargs) -> Module:
        return XeLU()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
