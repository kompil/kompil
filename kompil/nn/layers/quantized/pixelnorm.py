import torch
import torch.nn.functional

from torch.nn.modules.module import Module

from kompil.nn.layers.quantized.base import _Quantized
from kompil.nn.layers.pixelnorm import PixelNorm, pixelnorm


class ObservedPixelNorm(Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return pixelnorm(x)

    @classmethod
    def from_float(cls, float_pixelnorm: PixelNorm):
        obs_pixelnorm = cls(float_pixelnorm.eps)

        # qconfig has to be transmitted. If not, pytorch won't know how to quantize the thing.
        if hasattr(float_pixelnorm, "qconfig"):
            setattr(obs_pixelnorm, "qconfig", getattr(float_pixelnorm, "qconfig"))

        return obs_pixelnorm


class QuantizedPixelNorm(_Quantized):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x_dq = torch.dequantize(x)
        norm = pixelnorm(x_dq)
        x_q = torch.quantize_per_tensor(norm, self.scale, self.zero_point, dtype=x.dtype)

        return x_q

    @classmethod
    def from_observed(cls, obs_pixelnorm: ObservedPixelNorm):
        scale, zero_point = obs_pixelnorm.activation_post_process.calculate_qparams()

        quant_pixelnorm = cls(obs_pixelnorm.eps)
        quant_pixelnorm.scale = scale
        quant_pixelnorm.zero_point = zero_point

        return quant_pixelnorm

    def extra_repr(self) -> str:
        return f"scale={self.scale}, zero_point={self.zero_point}"
