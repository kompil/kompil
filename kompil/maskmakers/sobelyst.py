import torch
import torch.nn.functional as F

from kompil.maskmakers.factory import AutoMaskMakerBase, MASKTYPE, register_maskmaker
from kompil.utils.colorspace import colorspace_420_to_444, convert_to_colorspace
from kompil.utils.video import display_frame
from kornia.filters import sobel, gaussian_blur2d
from kornia.color import rgb_to_grayscale


def _hysteresis(i: torch.Tensor, coeff: float = 2.5):
    coeff_tensor = torch.ones_like(i) * coeff
    hysteresis_effect = torch.pow(coeff_tensor - 1, i - 1) / torch.pow(coeff_tensor, i)
    return torch.nan_to_num(hysteresis_effect, nan=0, posinf=0, neginf=0)


@register_maskmaker("sobelyst")
class MaskMakerSobelyst(AutoMaskMakerBase):
    def __init__(self, blur: bool = False, display: bool = False):
        self.__mask = None
        self.__stimuli = None
        self.__hysteresis = None
        self.__prev_edge_detect = None
        self.__display = display
        self.__blur = blur
        self.__counter = 0
        self.__mul = 5
        self.__hysteresis_coeff = 2.5

    def init(self, nb_frames: int, frame_shape: torch.Size):
        self.__mask = torch.zeros(nb_frames, *frame_shape)
        self.__stimuli = torch.zeros(nb_frames, *frame_shape)
        self.__hysteresis = _hysteresis(torch.arange(nb_frames), self.__hysteresis_coeff)

    def push_frame(self, frame: torch.Tensor):
        idx = self.__counter
        self.__counter += 1
        mask = self.__mask[idx]

        frame_rgb = torch.clamp(convert_to_colorspace(frame, "yuv420", "rgb8"), 0, 1.0)

        with torch.no_grad():
            # Edge detection via Sobel kernel
            edge_detect_grayscale = rgb_to_grayscale(
                sobel(frame_rgb.unsqueeze(0)).cpu() * self.__mul
            )

            if self.__blur:
                edge_detect_grayscale = gaussian_blur2d(edge_detect_grayscale, (7, 7), (2.5, 2.5))

            # Conversion to yuv with mask application to every channel
            _, _, h_2, w_2 = edge_detect_grayscale.shape
            h = int(h_2 / 2)
            w = int(w_2 / 2)
            y = F.pixel_unshuffle(edge_detect_grayscale, 2)
            u = F.interpolate(edge_detect_grayscale, (h, w), mode="bilinear")
            v = F.interpolate(edge_detect_grayscale, (h, w), mode="bilinear")
            edge_detect_yuv = torch.cat([y, u, v], 1)

            if idx > 0:
                # Historical stimilus is based on edge evolution / difference
                self.__stimuli[idx] = torch.abs(edge_detect_yuv - self.__prev_edge_detect)

                # Apply an hysteresis effect on old frames
                for i in range(idx):
                    mask += self.__stimuli[idx - i - 1] * self.__hysteresis[i]
            else:
                self.__stimuli[idx] = edge_detect_yuv
                mask = edge_detect_yuv * self.__hysteresis[0]

            self.__prev_edge_detect = edge_detect_yuv

            self.__mask[idx] = torch.clamp(mask, 0, 1)

        if not self.__display:
            return

        mask_yuv444 = colorspace_420_to_444(self.__mask[idx].unsqueeze(0)).squeeze(0)

        display_frame(mask_yuv444[0], "mask", 1)

    def compute(self) -> torch.HalfTensor:
        return self.__mask.to(MASKTYPE)
