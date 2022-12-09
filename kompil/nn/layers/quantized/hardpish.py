import torch

from torch.nn.modules.module import Module
from torch.nn.quantized import FloatFunctional, QFunctional
from torch.nn.functional import relu
from torch._ops import ops

from kompil.nn.layers.hardpish import BuilderHardPish, HardPish
from kompil.nn.layers.quantized.base import _Quantized


class ObservedHardPish(Module):
    def __init__(self):
        super().__init__()
        self.weight = None
        self.depth_neg_mul_ops = FloatFunctional()
        self.add_ops = FloatFunctional()

    @classmethod
    def from_float(cls, float_hardpish: HardPish):
        obs_hardpish = cls()
        obs_hardpish.weight = float_hardpish.weight

        # qconfig has to be transmitted. If not, pytorch won't know how to quantize the thing.
        if hasattr(float_hardpish, "qconfig"):
            setattr(obs_hardpish, "qconfig", getattr(float_hardpish, "qconfig"))

        return obs_hardpish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = relu(x)
        neg = -relu(-x)

        factor_pos = self.weight[0] * pos
        depth_neg = self.depth_neg_mul_ops.mul(neg, relu(neg + self.weight[3])) / self.weight[1]
        factor_neg = -relu(-self.weight[2] * neg)
        bias = self.weight[4]

        y = self.add_ops.add(factor_pos, depth_neg) + factor_neg + bias

        return y

    def extra_repr(self) -> str:
        return f"weight={self.weight}"


class QuantizedHardPish(_Quantized):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.Tensor(5))

        self.depth_neg_mul_ops = QFunctional()
        self.add_ops = QFunctional()

    @classmethod
    def from_observed(cls, obs_hardpish: ObservedHardPish):
        scale_mod, zp_mod = obs_hardpish.activation_post_process.calculate_qparams()

        quant_hardpish = cls()
        quant_hardpish.weight = obs_hardpish.weight
        quant_hardpish.scale = scale_mod
        quant_hardpish.zero_point = zp_mod

        neg_mul_act = obs_hardpish.depth_neg_mul_ops.activation_post_process
        scale_neg_mul, zp_neg_mul = neg_mul_act.calculate_qparams()

        quant_hardpish.depth_neg_mul_ops.scale = scale_neg_mul
        quant_hardpish.depth_neg_mul_ops.zero_point = zp_neg_mul

        add_act = obs_hardpish.add_ops.activation_post_process
        scale_add, zp_add = add_act.calculate_qparams()

        quant_hardpish.add_ops.scale = scale_add
        quant_hardpish.add_ops.zero_point = zp_add

        return quant_hardpish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = relu(x)
        neg = ops.quantized.mul_scalar(relu(ops.quantized.mul_scalar(x, -1)), -1)

        factor_pos = ops.quantized.mul_scalar(pos, self.weight[0])
        cliff_neg = ops.quantized.mul_scalar(
            relu(ops.quantized.add_scalar(neg, self.weight[3])), 1 / self.weight[1]
        )
        depth_neg = self.depth_neg_mul_ops.mul(neg, cliff_neg)
        factor_neg = ops.quantized.mul_scalar(
            relu(ops.quantized.mul_scalar(neg, -self.weight[2])), -1
        )

        first_add = self.add_ops.add(factor_pos, depth_neg)
        second_add = ops.quantized.add(first_add, factor_neg, self.scale, self.zero_point)
        y = ops.quantized.add_scalar(second_add, self.weight[4])

        return y

    def extra_repr(self) -> str:
        return f"weight={self.weight}"


class BuilderQuantizedHardPish:
    TYPE = "quantized_hardpish"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, input_size, **kwargs) -> Module:
        return QuantizedHardPish()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return BuilderHardPish.predict_size(input_size, **kwargs)
