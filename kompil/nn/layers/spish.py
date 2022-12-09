import torch

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import relu

from kompil.nn.layers.pish import pish


def spish(x, weight):
    pos = relu(x)
    neg = -relu(-x)
    pished_neg = pish(neg, weight)

    y = pished_neg + pos

    return y


class SPish(Module):
    def __init__(self):
        super().__init__()

        self.weight = Parameter(torch.Tensor([0.1, 1]))

    def forward(self, x):
        return spish(x, self.weight)
