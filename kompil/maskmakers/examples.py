import torch

from .factory import AutoMaskMakerBase, MASKTYPE, register_maskmaker

from kompil.utils.colorspace import colorspace_420_to_444, convert_to_colorspace
from kompil.utils.video import display_frame


@register_maskmaker("example-empty")
class MaskMakerEmpty(AutoMaskMakerBase):
    def __init__(self):
        self.__mask = None

    def init(self, nb_frames: int, frame_shape: torch.Size):
        self.__mask = torch.zeros(nb_frames, *frame_shape, dtype=MASKTYPE)

    def push_frame(self, frame: torch.Tensor):
        # Do nothing
        pass

    def compute(self) -> torch.HalfTensor:
        return self.__mask


@register_maskmaker("example-constants")
def mask_constants(frame: torch.Tensor, value: float = 1.0):
    return torch.ones_like(frame, dtype=torch.float16) * value


@register_maskmaker("example-display")
def mask_display(frame: torch.Tensor, display: bool = True):
    mask = torch.zeros_like(frame, dtype=torch.float16)

    if not display:
        return mask

    frame_rgb = torch.clamp(convert_to_colorspace(frame, "yuv420", "rgb8"), 0, 1.0)
    mask_yuv444 = colorspace_420_to_444(mask.unsqueeze(0)).squeeze(0)

    display_frame(frame_rgb, "original", 1)
    display_frame(mask_yuv444[0], "mask Y", 1)
    display_frame(mask_yuv444[1], "mask U", 1)
    display_frame(mask_yuv444[2], "mask V", 0)

    return mask
