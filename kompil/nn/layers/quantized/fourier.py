import torch
import torch.nn.functional

from torch.nn.modules.module import Module

from kompil.nn.layers.quantized.base import _Quantized
from kompil.nn.layers.fourier import BuilderFourierFeature, FourierFeature, fourier_feature


class ObservedFourierFeature(Module):
    def __init__(self, input_chan, output_chan):
        super().__init__()
        self.input_chan = input_chan
        self.output_chan = output_chan
        self.weight = None

    def forward(self, x):
        x = fourier_feature(x, self.weight)
        return x

    @classmethod
    def from_float(cls, float_ff: FourierFeature):
        obs_ff = cls(float_ff._input_chan, float_ff._mapping_size * 2)
        obs_ff.weight = float_ff.weight

        # qconfig has to be transmitted. If not, pytorch won't know how to quantize the thing.
        if hasattr(float_ff, "qconfig"):
            setattr(obs_ff, "qconfig", getattr(float_ff, "qconfig"))

        return obs_ff


class QuantizedFourierFeature(_Quantized):
    def __init__(self, input_chan: int, output_chan: int):
        super().__init__()
        self.register_buffer("weight", torch.Tensor(input_chan, output_chan // 2))

    def forward(self, x):
        x_dq = torch.dequantize(x)
        ff = fourier_feature(x_dq, self.weight)
        x_q = torch.quantize_per_tensor(ff, self.scale, self.zero_point, dtype=x.dtype)

        return x_q

    @classmethod
    def from_observed(cls, obs_ff: ObservedFourierFeature):
        scale, zero_point = obs_ff.activation_post_process.calculate_qparams()

        quant_ff = cls(obs_ff.input_chan, obs_ff.output_chan)
        quant_ff.weight = obs_ff.weight
        quant_ff.scale = scale
        quant_ff.zero_point = zero_point

        return quant_ff

    def extra_repr(self) -> str:
        return f"scale={self.scale}, zero_point={self.zero_point}"


class BuilderQuantizedFourierFeature:
    TYPE = "quantized_fourier_feature"

    @classmethod
    def elem(cls, output_chan: int) -> dict:
        return dict(type=cls.TYPE, output_chan=output_chan)

    @classmethod
    def make(cls, input_size, output_chan: int, **kwargs) -> Module:
        return QuantizedFourierFeature(input_size[0], output_chan)

    @classmethod
    def predict_size(cls, input_size, output_chan, **kwargs) -> tuple:
        return BuilderFourierFeature.predict_size(input_size, output_chan, **kwargs)
