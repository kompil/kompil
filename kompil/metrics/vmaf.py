import torch
import kornia
from typing import Optional

from kompil.metrics.metrics import Metric, MetricEmptyShell, register_metric

try:
    from kompil_vmaf import VmafCompute as _VmafCompute

    has_kompil_ext = True
except ImportError as exception:
    print("WARNING: kompil vmaf extension not imported. VMAF compute will be disabled.")
    has_kompil_ext = False

VMAF_0_6_1 = "vmaf_v0.6.1"


class VMAF(Metric):
    HIGHER_IS_BETTER = True

    def __init__(self, model_path: str):
        self.__compute = _VmafCompute(model_path)
        self.__count_frames = 0

    @staticmethod
    def __prep_image(img: torch.Tensor) -> torch.Tensor:
        """
        RGB8 -> YUV444 -> YUV444N -> [0,255] -> uint8 -> contiguous
        """
        yuv444 = kornia.color.yuv.rgb_to_yuv(img)

        yuv444n = torch.zeros_like(yuv444)

        yuv444n[0] = yuv444[0]
        yuv444n[1] = (yuv444[1] + 0.436) / 0.872
        yuv444n[2] = (yuv444[2] + 0.615) / 1.23

        yuv444n_255 = yuv444n * 255

        return yuv444n_255.to(torch.uint8).contiguous().cpu()

    @property
    def frame_count(self) -> int:
        return self.__compute.get_frame_count()

    def add_frame(self, ref: torch.Tensor, dist: torch.Tensor):
        ref_prep = self.__prep_image(ref)
        dist_prep = self.__prep_image(dist)
        self.__compute.add_frame(ref_prep, dist_prep)
        self.__count_frames += 1

    def compute(self) -> float:
        return self.__compute.flush()

    def pool_mean(self, index: Optional[int] = None) -> float:
        index = index if index is not None else self.__count_frames - 1
        return self.__compute.pool_mean(index)

    def score_at_index(self, index: int) -> float:
        return self.__compute.score_at_index(index)

    def get_score_list(self) -> torch.Tensor:
        scores = torch.empty(self.__compute.get_frame_count(), dtype=torch.float64)
        self.__compute.feed_scores(scores)
        return scores.to(torch.float32)


if not has_kompil_ext:
    VMAF = MetricEmptyShell

register_metric("vmaf", model_path=VMAF_0_6_1)(VMAF)
