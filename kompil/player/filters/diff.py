import torch
from typing import List

from kompil.utils.colorspace import convert_to_colorspace
from kompil.metrics import compute_pixel_dist, compute_pixel_ssim
from kompil.player.options import PlayerTargetOptions
from kompil.player.filters.filter import Filter, register_filter


def compute_pixel_dist_norm(frame1, frame2):
    euclidian_diff = compute_pixel_dist(frame1, frame2)
    euclidian_diff_min = euclidian_diff.min()
    euclidian_diff_max = euclidian_diff.max()
    return (euclidian_diff - euclidian_diff_min) / (euclidian_diff_max - euclidian_diff_min)


def compute_pixel_ssim_norm(frame1, frame2):
    ssim_diff = compute_pixel_ssim(frame1, frame2)
    ssim_diff_min = ssim_diff.min()
    ssim_diff_max = ssim_diff.max()
    return (ssim_diff - ssim_diff_min) / (ssim_diff_max - ssim_diff_min)


class DiffFilter(Filter):
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

    def filter(self, image: torch.Tensor, colorspace: str, idx: int) -> torch.Tensor:
        self.__decoder.set_position(idx)
        ref = self.__decoder.get_cur_frame()
        ref = convert_to_colorspace(ref, self.__decoder.get_colorspace(), "rgb8")
        image = convert_to_colorspace(image, colorspace, "rgb8")
        return self.compare(image, ref)

    def compare(self, frame1, frame2):
        raise NotImplementedError()


@register_filter("diff_pixel_euclidian")
class DiffPixelEuclidianFilter(DiffFilter):
    def compare(self, frame1, frame2):
        return compute_pixel_dist_norm(frame1, frame2)


@register_filter("diff_pixel_ssim")
class DiffPixelSSIMFilter(DiffFilter):
    def compare(self, frame1, frame2):
        return compute_pixel_ssim_norm(frame1, frame2)
