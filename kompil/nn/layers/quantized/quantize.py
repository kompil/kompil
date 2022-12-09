import torch

from kompil.nn.layers.quantized.base import _Quantized


class FixedQuantStub(torch.quantization.QuantStub):
    """
    Class to act like a QuantStub but replace the automatic approximation by a defined one.
    """

    def __init__(self, force_scale, force_zero_point, qconfig=None):
        super().__init__(qconfig=qconfig)
        self.force_scale = force_scale
        self.force_zero_point = force_zero_point

    def extra_repr(self) -> str:
        return f"scale={self.force_scale}, zero_point={self.force_zero_point}"


class Quantize(_Quantized):
    def __init__(self):
        super(Quantize, self).__init__()

        self.scale = 1
        self.zero_point = 0
        self.dtype = torch.quint8

    def forward(self, X):
        return torch.quantize_per_tensor(X, self.scale, self.zero_point, self.dtype)

    @staticmethod
    def from_float(mod):
        assert hasattr(mod, "activation_post_process")
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Quantize(
            scale.float().item(), zero_point.long().item(), mod.activation_post_process.dtype
        )

    def extra_repr(self):
        return "scale={}, zero_point={}, dtype={}".format(self.scale, self.zero_point, self.dtype)


class FixedQuantize(torch.nn.Module):
    """
    Class to act like a Quantize but use userly defined scale and zero point.
    """

    def __init__(self, scale: float, zero_point: int):
        super(FixedQuantize, self).__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, X):
        return torch.quantize_per_tensor(X, self.scale, self.zero_point, torch.quint8)

    def extra_repr(self):
        return f"scale={self.scale}, zero_point={self.zero_point}"
