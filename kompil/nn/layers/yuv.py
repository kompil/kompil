import torch

from torch.nn.modules.module import Module

from kompil.utils.colorspace import colorspace_420_to_444, DEFAULT_420_TO_444_MODE


class Yuv420ToYuv444(Module):
    def __init__(self, mode=DEFAULT_420_TO_444_MODE):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return colorspace_420_to_444(x, mode=self.mode)


class BuilderYuv420ToYuv444:
    TYPE = "colorspace_420_to_444"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, input_size, **kwargs) -> Module:
        return Yuv420ToYuv444()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        assert len(input_size) == 3
        assert input_size[0] == 6

        _, h_2, w_2 = input_size
        w, h = w_2 * 2, h_2 * 2
        output_size = (3, h, w)

        return output_size


class Yuv420astride(Module):
    def forward(self, x):
        x[:, 0:4] += 0.5
        return x


class BuilderYuv420astride:
    TYPE = "yuv420_astride"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, input_size, **kwargs) -> Module:
        return Yuv420astride()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
