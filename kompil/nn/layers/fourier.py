import torch

from torch.nn.modules.module import Module
from math import pi


def fourier_feature(x: torch.Tensor, weight: torch.Tensor):
    # Gaussian Fourier feature mapping.
    # https://arxiv.org/abs/2006.10739
    # https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    b, c, h, w = x.shape

    x = x.permute(0, 2, 3, 1).reshape(b * w * h, c)
    x = x @ weight.detach()
    x = x.view(b, h, w, weight.shape[1])
    x = x.permute(0, 3, 1, 2)
    x = 2 * pi * x
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)

    return x


class FourierFeature(Module):
    def __init__(self, input_chan: int, output_chan: int):
        super().__init__()

        assert output_chan % 2 == 0

        self._input_chan = input_chan
        self._mapping_size = output_chan // 2
        self.weight = torch.nn.Parameter(torch.randn((self._input_chan, self._mapping_size)) * 0.1)

    def forward(self, x):
        return fourier_feature(x, self.weight.detach())

    def extra_repr(self) -> str:
        return f"{self._input_chan}, {self._mapping_size * 2}"


class BuilderFourierFeature:
    TYPE = "fourier_feature"

    @classmethod
    def elem(cls, output_chan: int) -> dict:
        return dict(type=cls.TYPE, output_chan=output_chan)

    @classmethod
    def make(cls, input_size, output_chan: int, **kwargs) -> Module:
        return FourierFeature(input_size[0], output_chan)

    @classmethod
    def predict_size(cls, input_size, output_chan, **kwargs) -> tuple:
        return (output_chan, input_size[1], input_size[2])
