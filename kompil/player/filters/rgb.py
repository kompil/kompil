import torch
from typing import List

from kompil.utils.colorspace import convert_to_colorspace
from kompil.utils.video import resolution_to_chw
from kompil.player.filters.filter import Filter, register_filter
from kompil.player.options import PlayerTargetOptions


@register_filter("rgb")
class RGBFilter(Filter):
    @classmethod
    def generate(cls, data: List[str], opt: PlayerTargetOptions) -> Filter:
        assert len(data) == 0
        return cls(opt)

    def __init__(self, opt: PlayerTargetOptions):
        super().__init__()
        self.__resize_to = None
        if opt.resolution is not None:
            self.__resize_to = tuple(resolution_to_chw(opt.resolution)[1:3])
            self.__resize_to = self.__resize_to if self.__resize_to != (-1, -1) else None

    def filter(self, image: torch.Tensor, colorspace: str, idx: int) -> torch.Tensor:
        image = image.unsqueeze(0)
        rgb_image = convert_to_colorspace(image, colorspace, "rgb8", lossy_allowed=True)
        rgb_image.clip_(0.0, 1.0)

        if self.__resize_to is not None:
            rgb_image = torch.nn.functional.interpolate(
                rgb_image, size=self.__resize_to, mode="bilinear"
            )

        return rgb_image[0]
