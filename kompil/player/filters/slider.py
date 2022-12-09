import torch
from typing import List

from kompil.utils.colorspace import convert_to_colorspace
from kompil.player.options import PlayerTargetOptions
from kompil.player.filters.filter import Filter, register_filter


@register_filter("slider")
class SliderFilter(Filter):
    @classmethod
    def generate(cls, data: List[str], opt: PlayerTargetOptions) -> Filter:
        return cls(data, opt)

    def __init__(self, data: List[str], opt: PlayerTargetOptions):
        super().__init__()
        if len(data) == 1:
            decoder = "auto"
            fpath = data[0]
        else:
            decoder = data[0]
            fpath = data[1]
        from kompil.player.decoders.decoder import create_decoder

        self.__decoder = create_decoder(decoder, fpath, opt)
        self.__ratio = 0.5

    def filter(self, image: torch.Tensor, colorspace: str, idx: int) -> torch.Tensor:
        # RGB image
        image = convert_to_colorspace(image, colorspace, "rgb8")
        image.clip_(0.0, 1.0)
        # Get reference
        self.__decoder.set_position(idx)
        ref = self.__decoder.get_cur_frame()
        ref = convert_to_colorspace(ref, self.__decoder.get_colorspace(), "rgb8")
        # Make the split
        split_pixel = int((image.shape[2] - 1) * self.__ratio)
        image_part = image[:, :, :split_pixel]
        ref_part = ref[:, :, split_pixel + 1 :]
        bar = torch.zeros(3, image.shape[1], 1, dtype=image.dtype, device=image.device)
        # Concat
        return torch.cat([image_part, bar, ref_part], dim=2)

    def set_slider(self, ratio: float):
        self.__ratio = ratio

    @property
    def is_slider(self) -> bool:
        return True
