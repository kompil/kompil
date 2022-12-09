import torch
import torch.nn.functional

from kompil.nn.layers.quantized.base import _Quantized

from torch.nn.modules.module import Module
from torch._ops import ops


class ObservedPReLU(Module):
    def __init__(self):
        super().__init__()
        self.weight = None

    def forward(self, x):
        pos = torch.relu(x)
        neg = -self.weight * torch.relu(-x)
        y = pos + neg

        return y

    @classmethod
    def from_float(cls, float_prelu: torch.nn.PReLU):
        obs_prelu = cls()
        obs_prelu.weight = float_prelu.weight

        # qconfig has to be transmitted. If not, pytorch won't know how to quantize the thing.
        if hasattr(float_prelu, "qconfig"):
            setattr(obs_prelu, "qconfig", getattr(float_prelu, "qconfig"))

        return obs_prelu

    def extra_repr(self) -> str:
        return f"negative_slop={self.weight.item()}"


class QuantizedPReLU(_Quantized):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.Tensor(1))

    def forward(self, x):
        pos = torch.relu(x)
        neg = ops.quantized.mul_scalar(torch.relu(ops.quantized.mul_scalar(x, -1)), -self.weight)
        y = ops.quantized.add(pos, neg, self.scale, self.zero_point)

        return y

    @classmethod
    def from_observed(cls, obs_prelu: ObservedPReLU):
        scale, zero_point = obs_prelu.activation_post_process.calculate_qparams()

        quant_prelu = cls()
        quant_prelu.weight = obs_prelu.weight
        quant_prelu.scale = scale
        quant_prelu.zero_point = zero_point

        return quant_prelu

    def extra_repr(self) -> str:
        return (
            f"negative_slop={self.weight.item()}, scale={self.scale}, zero_point={self.zero_point}"
        )
