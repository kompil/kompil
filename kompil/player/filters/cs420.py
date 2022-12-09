import numpy
import torch
from typing import List

from kompil.utils.video import tensor_to_numpy
from kompil.utils.colorspace import ycbcr420shift_to_ycbcr420
from kompil.player.options import PlayerTargetOptions
from kompil.player.filters.filter import Filter, register_filter


@register_filter("cs420")
class CS420Filter(Filter):
    @classmethod
    def generate(cls, data: List[str], opt: PlayerTargetOptions) -> Filter:
        assert len(data) == 0
        return cls()

    def __init__(self):
        super().__init__()

    def filter(self, image: torch.Tensor, colorspace: str, idx: int) -> torch.Tensor:
        if colorspace == "yuv420":
            return self.__filter_yuv420(image)
        if colorspace == "ycbcr420":
            return self.__filter_ycbcr420(image)
        if colorspace == "ycbcr420shift":
            return self.__filter_ycbcr420shift(image)
        raise RuntimeError(f"Colorspace {colorspace} not handled by filter cs420")

    def __stack(self, i0, i1, i2) -> torch.Tensor:
        raise NotImplementedError()
        i0 = tensor_to_numpy(i0)
        i1 = tensor_to_numpy(i1)
        i2 = tensor_to_numpy(i2)
        return numpy.hstack((i0, numpy.vstack((i1, i2))))

    def __filter_ycbcr420shift(self, image: torch.Tensor) -> torch.Tensor:
        return self.__filter_ycbcr420(ycbcr420shift_to_ycbcr420(image.unsqueeze(0))[0])

    def __filter_ycbcr420(self, image: torch.Tensor) -> torch.Tensor:
        y = torch.pixel_shuffle(image[0:4], 2)
        cb = image[4:5]
        cr = image[5:6]
        return self.__stack(y, cb, cr)

    def __filter_yuv420(self, image: torch.Tensor) -> torch.Tensor:
        y = torch.pixel_shuffle(image[0:4], 2)
        u = image[4:5]
        v = image[5:6]

        y.clamp_(0.0, 1.0)
        u.clamp_(-0.436, 0.436)
        u.clamp_(-0.615, 0.615)

        u = (u + 0.436) / (2 * 0.436)
        v = (v + 0.615) / (2 * 0.615)

        return self.__stack(y, u, v)
