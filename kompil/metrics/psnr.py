import torch
import kornia
from typing import Optional

from kompil.metrics.metrics import Metric, register_metric


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0):
    """
    Compute the PSNR between two images.
    """
    mse = torch.mean((img1 - img2) ** 2)

    return 20 * torch.log10(max_val / torch.sqrt(mse))


@register_metric("psnr")
class PSNR(Metric):
    HIGHER_IS_BETTER = True

    def __init__(self, max_val: float = 1.0):
        self.__max_val = max_val
        self.__scores = []

    @property
    def frame_count(self) -> int:
        return len(self.__scores)

    def add_frame(self, ref: torch.Tensor, dist: torch.Tensor):
        self.__scores.append(compute_psnr(ref, dist, max_val=self.__max_val).item())

    def compute(self) -> float:
        pass

    def score_at_index(self, index: int) -> float:
        return self.__scores[index]

    def get_score_list(self) -> torch.Tensor:
        return torch.Tensor(self.__scores).to(torch.float32)
