import torch
import math

from torch import Tensor
from torch.nn.modules.module import Module
from torch.autograd import Function


class _FunctionClamp(Function):
    @staticmethod
    def forward(ctx, input: Tensor, bound: float) -> Tensor:
        return torch.clamp(input, -math.inf, bound)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output, None


class PReLUB(Module):
    def __init__(self, bound: int):
        super().__init__()

        self._prelu = torch.nn.PReLU()
        self._bound = bound

    def forward(self, x: Tensor) -> Tensor:
        x_prelu = self._prelu(x)
        x_clamped = _FunctionClamp.apply(x_prelu, self._bound)

        return x_clamped

    def extra_repr(self) -> str:
        return f"bound={self._bound}"


class BuilderPReLUB:
    TYPE = "prelub"

    @classmethod
    def elem(cls, bound: int = 6) -> dict:
        return dict(type=cls.TYPE, bound=bound)

    @classmethod
    def make(cls, bound: int, **kwargs) -> Module:
        return PReLUB(bound)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
