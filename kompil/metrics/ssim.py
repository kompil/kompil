import torch
from typing import Optional

from kompil.metrics.metrics import Metric, register_metric
from kornia.losses.ssim import ssim as kornia_ssim


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0):
    """
    Compute the SSIM between two images.
    """
    if len(img1.shape) == 3:  # chw to nchw (n=1)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    ssim_val = kornia_ssim(img1, img2, window_size=7, max_val=max_val)

    return ssim_val.mean()  # 1 - ssim_val * 2  # SSIM loss is based on DSSSIM = (SSIM - 1) / 2


def compute_pixel_ssim(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0):
    if len(img1.shape) == 3:  # chw to nchw (n=1)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    ssim_val = kornia_ssim(img1, img2, window_size=7, max_val=max_val)

    if len(ssim_val.shape) == 4:  # nchw to chw
        ssim_val = ssim_val.squeeze(0)

    ssim_pixel_mean = torch.mean(ssim_val, dim=-3, keepdim=True)

    return ssim_pixel_mean


@register_metric("ssim")
class SSIM(Metric):
    HIGHER_IS_BETTER = True

    def __init__(self, max_val: float = 1.0):
        self.__max_val = max_val
        self.__scores = []

    @property
    def frame_count(self) -> int:
        return len(self.__scores)

    def add_frame(self, ref: torch.Tensor, dist: torch.Tensor):
        self.__scores.append(compute_ssim(ref, dist, max_val=self.__max_val).item())

    def compute(self) -> float:
        pass

    def score_at_index(self, index: int) -> float:
        return self.__scores[index]

    def get_score_list(self) -> torch.Tensor:
        return torch.Tensor(self.__scores).to(torch.float32)
