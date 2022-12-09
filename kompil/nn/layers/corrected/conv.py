import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict
from torch._ops import ops

from kompil.nn.layers.corrected.corrections import _Correction
from kompil.nn.layers.corrected.base import _Corrected


def _reverse_repeat_padding(padding: List[int]) -> List[int]:
    _reversed_padding_repeated_twice: List[int] = []
    N = len(padding)

    for idx in range(N):
        for _ in range(2):
            _reversed_padding_repeated_twice.append(padding[N - idx - 1])

    return _reversed_padding_repeated_twice


class CorrectedConv2d(nn.quantized.Conv2d, _Corrected):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        output_padding=0,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super(CorrectedConv2d, self)._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            output_padding,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.scale = 1.0
        self.zero_point = 0
        self.correction = None

    def _get_name(self):
        return "CorrectedConv2d"

    @classmethod
    def from_quantized(cls, q_conv2d: nn.quantized.Conv2d, correction: _Correction, context: Dict):
        c_conv2d = cls(
            q_conv2d.in_channels,
            q_conv2d.out_channels,
            q_conv2d.kernel_size,
            q_conv2d.stride,
            q_conv2d.padding,
            q_conv2d.dilation,
            q_conv2d.output_padding,
            q_conv2d.groups,
            q_conv2d.bias is not None,
            q_conv2d.padding_mode,
        )

        c_conv2d.scale = q_conv2d.scale
        c_conv2d.zero_point = q_conv2d.zero_point
        c_conv2d._packed_params = q_conv2d._packed_params
        c_conv2d.context = context
        c_conv2d.correction = correction

        return c_conv2d

    def forward(self, x):
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            x = F.pad(x, _reversed_padding_repeated_twice, mode=self.padding_mode)

        convoluted = ops.quantized.conv2d(x, self._packed_params, self.scale, self.zero_point)
        corrected_convol = self._apply_correction(convoluted)

        return corrected_convol
