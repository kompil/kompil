import torch
import torch.nn.functional

from torch.nn.quantized import QFunctional
from kornia.color.yuv import YuvToRgb

from kompil.nn.layers.quantized.base import _Quantized
from kompil.utils.colorspace import quant_yuv444_to_rgb


class QuantizedYuv444ToRgb(_Quantized):
    def __init__(self):
        super().__init__()
        self.quant_ops = QFunctional()

    def forward(self, x):
        return quant_yuv444_to_rgb(x, self.quant_ops)

    @classmethod
    def from_float(cls, float_yuv: YuvToRgb):
        # Original YUVconv layer will act like the observer.
        obs_yuv = float_yuv

        # qconfig has to be transmitted. If not, pytorch won't know how to quantize the thing.
        if hasattr(float_yuv, "qconfig"):
            setattr(obs_yuv, "qconfig", getattr(float_yuv, "qconfig"))

        return obs_yuv

    @classmethod
    def from_observed(cls, obs_yuv: YuvToRgb):
        scale, zero_point = obs_yuv.activation_post_process.calculate_qparams()

        quant_yuv = cls()
        quant_yuv.quant_ops.scale = scale
        quant_yuv.quant_ops.zero_point = zero_point

        return quant_yuv
