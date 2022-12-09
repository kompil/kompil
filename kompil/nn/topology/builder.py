import torch
import math
import copy

from typing import Tuple, Union, List, Optional
from torch.nn import Sequential
from torch.nn import Module

import kornia.color.hsv
import kompil.nn.layers as layer
import kompil.nn.layers.quantized as qlayer
import kompil.nn.layers.corrected as clayer

from kompil.utils.factory import Factory


def _make_2d_tuple(value):
    if isinstance(value, int):
        return (value, value)
    elif isinstance(value, list):
        return tuple(value)
    return value


def _make_3d_tuple(value):
    if isinstance(value, int):
        return (value, value, value)
    elif isinstance(value, list):
        return tuple(value)
    return value


def _single_to_int(value: Union[int, Tuple[int], List[int]]):
    if isinstance(value, int):
        return value
    assert len(value) == 1, "_single_to_int only for single values"
    return value[0]


def _to_tuple(value: Union[int, Tuple[int], List[int]]):
    if isinstance(value, int):
        return (value,)
    return tuple(value)


__FACTORY = Factory("layers")


def layers_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_layer(cls):
    """
    Register the defined class as a layer.
    """
    return layers_factory().register(cls.TYPE)(cls)


@register_layer
class Save:
    TYPE = "save"

    @classmethod
    def elem(cls, sequence, path: str) -> dict:
        assert path

        if not isinstance(sequence, list):
            sequence = [sequence]

        return dict(type=cls.TYPE, sequence=sequence, id=path)

    @classmethod
    def make(cls, input_size, sequence, id, context, **kwargs) -> Module:
        path = id
        t, output_size = build_topology_from_list(
            sequence=sequence, input_size=input_size, context=context
        )
        return layer.SaveModule(path=id, sequence=t, sequence_list=sequence)

    @classmethod
    def predict_size(cls, input_size, sequence, **kwargs) -> tuple:
        return predict_size(sequence=sequence, input_size=input_size)


@register_layer
class Load:
    TYPE = "load"

    @classmethod
    def elem(cls, sequence, path: str, learnable: bool = False) -> dict:
        assert path

        if not isinstance(sequence, list):
            sequence = [sequence]

        return dict(type=cls.TYPE, sequence=sequence, id=path, mode=learnable)

    @classmethod
    def make(cls, input_size, sequence, mode, id, context, **kwargs) -> Module:
        path = id
        learnable = mode

        t, output_size = build_topology_from_list(
            sequence=sequence, input_size=input_size, context=context
        )
        return layer.LoadModule(path=id, sequence=t, sequence_list=sequence, learnable=learnable)

    @classmethod
    def predict_size(cls, input_size, sequence, **kwargs) -> tuple:
        return predict_size(sequence=sequence, input_size=input_size)


@register_layer
class Prune:
    TYPE = "prune"

    @classmethod
    def elem(cls, module, names=("weight", "bias")) -> dict:

        if not isinstance(module, list):
            module = [module]

        return dict(type=cls.TYPE, module=module, names=names)

    @classmethod
    def make(cls, input_size, module, names, context, **kwargs) -> Module:
        t, output_size = build_topology_from_list(
            sequence=module, input_size=input_size, context=context
        )
        return layer.Prune(module=t, names=names)

    @classmethod
    def predict_size(cls, input_size, module, **kwargs) -> tuple:
        return predict_size(sequence=module, input_size=input_size)


@register_layer
class Sequence:
    TYPE = "sequence"

    @classmethod
    def elem(cls, sequence) -> dict:
        if not isinstance(sequence, list):
            sequence = [sequence]
        return dict(type=cls.TYPE, sequence=sequence)

    @classmethod
    def make(cls, input_size, sequence, context, **kwargs) -> Module:
        built_seq, _ = build_topology_from_list(sequence, input_size, context)
        return torch.nn.Sequential(built_seq._modules)

    @classmethod
    def predict_size(cls, input_size, sequence, **kwargs) -> tuple:
        return predict_size(sequence=sequence, input_size=input_size)


@register_layer
class Quantize:
    TYPE = "quantize"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, input_size, **kwargs) -> Module:
        return qlayer.Quantize()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class FixedQuantize:
    TYPE = "fixed_quantize"

    @classmethod
    def elem(cls, scale: float, zero_point: int) -> dict:
        return dict(type=cls.TYPE, scale=scale, zero_point=zero_point)

    @classmethod
    def make(cls, scale, zero_point, **kwargs) -> Module:
        return qlayer.FixedQuantize(scale, zero_point)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class DeQuantize:
    TYPE = "dequantize"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, input_size, **kwargs) -> Module:
        return qlayer.DeQuantize()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class QuantizeStub:
    TYPE = "quantize_stub"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, input_size, **kwargs) -> Module:
        return torch.quantization.QuantStub()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class FixedQuantizeStub:
    TYPE = "fixed_quantstub"

    @classmethod
    def elem(cls, scale: float, zero_point: int) -> dict:
        return dict(type=cls.TYPE, scale=scale, zero_point=zero_point)

    @classmethod
    def make(cls, scale, zero_point, **kwargs) -> Module:
        return qlayer.FixedQuantStub(scale, zero_point)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class DeQuantizeStub:
    TYPE = "dequantize_stub"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, input_size, **kwargs) -> Module:
        return torch.quantization.DeQuantStub()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class Mish:
    TYPE = "mish"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return torch.nn.Mish()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class SPish:
    TYPE = "spish"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return layer.SPish()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class QuantizedSPish:
    TYPE = "quantized_spish"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return qlayer.QuantizedSPish()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class Pish:
    TYPE = "pish"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return layer.Pish()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class QuantizedPish:
    TYPE = "quantized_pish"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return qlayer.QuantizedPish()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return Pish.predict_size(input_size, **kwargs)


@register_layer
class Wish:
    TYPE = "wish"

    @classmethod
    def elem(cls, gate: float = -1) -> dict:
        return dict(type=cls.TYPE, gate=gate)

    @classmethod
    def make(cls, gate, **kwargs) -> Module:
        return layer.Wish(gate)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class ReLU:
    TYPE = "relu"

    @classmethod
    def elem(cls, inplace: bool = False) -> dict:
        return dict(type=cls.TYPE, inplace=inplace)

    @classmethod
    def make(cls, inplace, **kwargs) -> Module:
        return torch.nn.ReLU(inplace)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class LeakyReLU:
    TYPE = "lrelu"

    @classmethod
    def elem(cls, alpha: float = 0.01, inplace: bool = False) -> dict:
        return dict(type=cls.TYPE, alpha=alpha, inplace=inplace)

    @classmethod
    def make(cls, alpha, inplace, **kwargs) -> Module:
        return torch.nn.LeakyReLU(alpha, inplace)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class PReLU:
    TYPE = "prelu"

    @classmethod
    def elem(cls, nb_params: int) -> dict:
        return dict(type=cls.TYPE, nb_params=nb_params)

    @classmethod
    def single(cls) -> dict:
        return cls.elem(1)

    @classmethod
    def multi(cls) -> dict:
        return cls.elem(-1)

    @classmethod
    def make(cls, input_size, nb_params=1, **kwargs) -> Module:
        if nb_params != 1:
            nb_params = input_size

            if isinstance(nb_params, tuple) and len(nb_params) == 3:
                nb_params = nb_params[0]

        return torch.nn.PReLU(num_parameters=nb_params)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class QuantizedPReLU:
    TYPE = "quantized_prelu"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, input_size, **kwargs) -> Module:
        return qlayer.QuantizedPReLU()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return PReLU.predict_size(input_size, **kwargs)


@register_layer
class Softmax:
    TYPE = "softmax"

    @classmethod
    def elem(cls, dim: bool = None) -> dict:
        return dict(type=cls.TYPE, dim=dim)

    @classmethod
    def make(cls, dim, **kwargs) -> Module:
        dim = None if dim is None else dim + 1
        return torch.nn.Softmax(dim)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class Linear:
    TYPE = "linear"

    @classmethod
    def elem(cls, output_size: Union[int, str], bias: bool = True) -> dict:
        return dict(type=cls.TYPE, output_size=_single_to_int(output_size), bias=bias)

    @classmethod
    def make(cls, input_size, output_size, bias, **kwargs) -> Module:
        return torch.nn.Linear(_single_to_int(input_size), output_size, bias=bias)

    @classmethod
    def predict_size(cls, output_size, **kwargs) -> tuple:
        return (output_size,)


@register_layer
class QuantizedLinear:
    TYPE = "quantized_linear"

    @classmethod
    def elem(cls, output_size: Union[int, str], bias: bool) -> dict:
        return dict(
            type=cls.TYPE,
            output_size=_single_to_int(output_size),
            bias=bias,
        )

    @classmethod
    def make(cls, input_size, output_size, bias, **kwargs) -> Module:
        qlinear = torch.nn.quantized.Linear(_single_to_int(input_size), output_size)

        return qlinear

    @classmethod
    def predict_size(cls, output_size, **kwargs) -> tuple:
        return Linear.predict_size(output_size, **kwargs)


@register_layer
class Adjacent1d:
    TYPE = "adjacent1d"

    @classmethod
    def elem(
        cls,
        kernel: int,
        output_chan: Union[int, str],
        output_size: Union[int, str],
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=kernel,
            output_chan=output_chan,
            output_size=output_size,
            bias=bias,
        )

    @classmethod
    def make(
        cls,
        input_size,
        output_size,
        output_chan,
        bias,
        kernel,
        **kwargs,
    ) -> Module:
        return layer.Adjacent1d(
            input_dim=input_size,
            output_dim=(output_chan, output_size),
            kernel_size=kernel,
            bias=bias,
        )

    @classmethod
    def predict_size(cls, output_size, output_chan, **kwargs) -> tuple:
        return (output_chan, output_size)


@register_layer
class Adjacent2d:
    TYPE = "adjacent2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str],
        output_size: Union[int, str],
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_2d_tuple(kernel),
            output_size=output_size,
            bias=bias,
            output_channels=output_chan,
        )

    @classmethod
    def make(
        cls,
        input_size,
        output_size,
        output_channels,
        bias,
        kernel,
        **kwargs,
    ) -> Module:
        return layer.Adjacent2d(
            input_dim=input_size,
            output_dim=(output_channels, *output_size),
            kernel_size=kernel,
            bias=bias,
            acceleration=layer.Method.CUDA,
        )

    @classmethod
    def predict_size(cls, output_size, output_channels, **kwargs) -> tuple:
        return (output_channels, *output_size)


@register_layer
class BatchNorm1d:
    TYPE = "batchnorm1d"

    @classmethod
    def elem(cls, eps: float = 1e-5, momentum: float = 0.1) -> dict:
        return dict(type=cls.TYPE, epsilon=eps, momentum=momentum)

    @classmethod
    def make(cls, input_size, epsilon, momentum, **kwargs) -> Module:
        return torch.nn.BatchNorm1d(input_size, eps=epsilon, momentum=momentum)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class BatchNorm2d:
    TYPE = "batchnorm2d"

    @classmethod
    def elem(cls, eps: float = 1e-5, momentum: float = 0.1) -> dict:
        return dict(type=cls.TYPE, epsilon=eps, momentum=momentum)

    @classmethod
    def make(cls, input_size, epsilon, momentum, **kwargs) -> Module:
        return torch.nn.BatchNorm2d(input_size[0], eps=epsilon, momentum=momentum)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class Reshape:
    TYPE = "reshape"

    @classmethod
    def elem(cls, shape: tuple) -> dict:
        return dict(type=cls.TYPE, output_size=shape)

    @classmethod
    def make(cls, output_size, **kwargs) -> Module:
        return layer.Reshape(output_size)

    @classmethod
    def predict_size(cls, input_size, output_size, **kwargs) -> tuple:
        # Check dimensions match
        if isinstance(input_size, int):
            input_size = (input_size,)
        in_numel = 1
        for dim in input_size:
            in_numel *= dim
        if isinstance(output_size, int):
            output_size = (output_size,)
        out_numel = 1
        for dim in output_size:
            out_numel *= dim
        assert (
            out_numel == in_numel
        ), f"Reshape dimensions are not matching: {in_numel} != {out_numel}"
        # Direct output size
        return output_size


@register_layer
class PixelNorm:
    TYPE = "pixelnorm"

    @classmethod
    def elem(cls, eps=1e-8) -> dict:
        return dict(type=cls.TYPE, epsilon=eps)

    @classmethod
    def make(cls, epsilon, **kwargs) -> Module:
        return layer.PixelNorm(epsilon)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class QuantizedPixelNorm:
    TYPE = "quantized_pixelnorm"

    @classmethod
    def elem(cls, eps) -> dict:
        return dict(type=cls.TYPE, epsilon=eps)

    @classmethod
    def make(cls, epsilon, **kwargs) -> Module:
        return qlayer.QuantizedPixelNorm(epsilon)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class Conv1d:
    TYPE = "conv1d"

    @classmethod
    def elem(
        cls,
        kernel: int,
        output_chan: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        padding_mode: str = "zeros",
        bias: bool = True,
        groups=1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=kernel,
            output_channels=output_chan,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
            groups=groups,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding_mode,
        bias,
        output_channels,
        padding,
        kernel,
        groups,
        **kwargs,
    ) -> Module:

        input_channel = input_size[0]

        module = torch.nn.Conv1d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            groups=groups,
        )

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, dilation, padding, kernel, output_channels, **kwargs
    ) -> tuple:
        output_s = math.floor(
            (input_size[1] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        )

        return (output_channels, output_s)


@register_layer
class Conv2d:
    TYPE = "conv2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str] = 1,
        stride: Union[int, tuple] = (1, 1),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
        groups=1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_2d_tuple(kernel),
            output_channels=output_chan,
            stride=_make_2d_tuple(stride),
            dilation=_make_2d_tuple(dilation),
            padding=_make_2d_tuple(padding),
            padding_mode=padding_mode,
            bias=bias,
            groups=groups,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding_mode,
        bias,
        output_channels,
        padding,
        kernel,
        groups,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.Conv2d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            groups=groups,
        )

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, dilation, padding, kernel, output_channels, **kwargs
    ) -> tuple:
        output_h = math.floor(
            (input_size[1] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1
        )
        output_w = math.floor(
            (input_size[2] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
        )

        return (output_channels, output_h, output_w)


@register_layer
class QuantizedConv2d:
    TYPE = "quantized_conv2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str] = 1,
        stride: Union[int, tuple] = (1, 1),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
        groups=1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_2d_tuple(kernel),
            output_channels=output_chan,
            stride=_make_2d_tuple(stride),
            dilation=_make_2d_tuple(dilation),
            padding=_make_2d_tuple(padding),
            padding_mode=padding_mode,
            bias=bias,
            groups=groups,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding_mode,
        bias,
        output_channels,
        padding,
        kernel,
        groups,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.quantized.Conv2d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            groups=groups,
        )

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, dilation, padding, kernel, output_channels, **kwargs
    ) -> tuple:
        return Conv2d.predict_size(
            input_size, stride, dilation, padding, kernel, output_channels, **kwargs
        )


@register_layer
class CorrectedConv2d:
    TYPE = "corrected_conv2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str] = 1,
        stride: Union[int, tuple] = (1, 1),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
        groups=1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_2d_tuple(kernel),
            output_channels=output_chan,
            stride=_make_2d_tuple(stride),
            dilation=_make_2d_tuple(dilation),
            padding=_make_2d_tuple(padding),
            padding_mode=padding_mode,
            bias=bias,
            groups=groups,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding_mode,
        bias,
        output_channels,
        padding,
        kernel,
        groups,
        context,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = clayer.CorrectedConv2d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            groups=groups,
        )
        module.context = context

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, dilation, padding, kernel, output_channels, **kwargs
    ) -> tuple:
        return Conv2d.predict_size(
            input_size, stride, dilation, padding, kernel, output_channels, **kwargs
        )


@register_layer
class Conv3d:
    TYPE = "conv3d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str] = 1,
        stride: Union[int, tuple] = (1, 1, 1),
        dilation: Union[int, tuple] = (1, 1, 1),
        padding: Union[int, tuple] = (0, 0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
        groups=1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_3d_tuple(kernel),
            output_channels=output_chan,
            stride=_make_3d_tuple(stride),
            dilation=_make_3d_tuple(dilation),
            padding=_make_3d_tuple(padding),
            padding_mode=padding_mode,
            bias=bias,
            groups=groups,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding_mode,
        bias,
        output_channels,
        padding,
        kernel,
        groups,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.Conv3d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            groups=groups,
        )

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, dilation, padding, kernel, output_channels, **kwargs
    ) -> tuple:
        output_d = math.floor(
            (input_size[1] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1
        )
        output_h = math.floor(
            (input_size[2] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
        )
        output_w = math.floor(
            (input_size[3] + 2 * padding[2] - dilation[2] * (kernel[2] - 1) - 1) / stride[2] + 1
        )

        return (output_channels, output_d, output_h, output_w)


@register_layer
class MaxPool2d:
    TYPE = "maxpool2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        stride: Union[int, tuple] = (0, 0),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        ceil_mode: bool = False,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_2d_tuple(kernel),
            stride=_make_2d_tuple(stride),
            dilation=_make_2d_tuple(dilation),
            padding=_make_2d_tuple(padding),
            ceil_mode=ceil_mode,
        )

    @classmethod
    def make(
        cls,
        stride,
        dilation,
        padding,
        kernel,
        ceil_mode,
        **kwargs,
    ) -> Module:
        module = torch.nn.MaxPool2d(
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=False,
        )

        return module

    @classmethod
    def predict_size(
        cls, input_size, stride, dilation, padding, kernel, output_channels, **kwargs
    ) -> tuple:
        output_h = math.floor(
            (input_size[1] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1
        )
        output_w = math.floor(
            (input_size[2] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
        )

        return (output_channels, output_h, output_w)


@register_layer
class AvgPool2d:
    TYPE = "avgpool2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        stride: Union[int, tuple] = (0, 0),
        padding: Union[int, tuple] = (0, 0),
        count_include_pad: bool = True,
        ceil_mode: bool = False,
        divisor_override: bool = None,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_2d_tuple(kernel),
            stride=_make_2d_tuple(stride),
            padding=_make_2d_tuple(padding),
            ceil_mode=ceil_mode,
            padding_mode=count_include_pad,
            divise_factor=divisor_override,
        )

    @classmethod
    def make(
        cls,
        stride,
        padding,
        kernel,
        ceil_mode,
        divise_factor,
        padding_mode,
        **kwargs,
    ) -> Module:
        divisor = divise_factor
        include_padding = padding_mode

        module = torch.nn.AvgPool2d(
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            divisor_override=divisor,
            ceil_mode=ceil_mode,
            count_include_pad=include_padding,
        )

        return module

    @classmethod
    def predict_size(cls, input_size, stride, padding, kernel, output_channels, **kwargs) -> tuple:
        output_h = math.floor((input_size[1] + 2 * padding[0] - kernel[0]) / stride[0] + 1)
        output_w = math.floor((input_size[2] + 2 * padding[1] - kernel[1]) / stride[1] + 1)

        return (output_channels, output_h, output_w)


@register_layer
class UpSample:
    TYPE = "upsample"

    @classmethod
    def elem(
        cls,
        output_size: tuple = None,
        scale_factor: Union[int, tuple] = (2, 2),
        mode: str = "bilinear",
        align_corners: bool = False,
    ) -> dict:
        if output_size:
            scale_factor = None
        elif isinstance(scale_factor, int):
            scale_factor = _make_2d_tuple(scale_factor)

        return dict(
            type=cls.TYPE,
            output_size=output_size,
            scale_factor=scale_factor,
            mode=mode,
            align_corner=align_corners,
        )

    @classmethod
    def make(cls, input_size, output_size, scale_factor, mode, align_corner, **kwargs) -> Module:
        return torch.nn.Upsample(
            size=output_size, scale_factor=scale_factor, mode=mode, align_corners=align_corner
        )

    @classmethod
    def predict_size(cls, input_size, output_size, scale_factor, **kwargs) -> tuple:

        if output_size:
            return (input_size[0], output_size[0], output_size[1])

        return (input_size[0], input_size[1] * scale_factor[0], input_size[2] * scale_factor[1])


@register_layer
class Deconv1d:
    TYPE = "deconv1d"

    @classmethod
    def elem(
        cls,
        kernel: int,
        output_chan: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        padding_mode: str = "zeros",
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=padding_mode,
            output_channels=output_chan,
            output_padding=output_padding,
            bias=bias,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        padding_mode,
        bias,
        kernel,
        output_channels,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.ConvTranspose1d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

        return module

    @classmethod
    def predict_size(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        kernel,
        output_channels,
        output_padding,
        **kwargs,
    ) -> tuple:
        output_s = (
            (input_size[1] - 1) * stride
            - 2 * padding
            + dilation * (kernel - 1)
            + output_padding
            + 1
        )

        return (output_channels, output_s)


@register_layer
class Deconv2d:
    TYPE = "deconv2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str] = 1,
        stride: Union[int, tuple] = (1, 1),
        dilation: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        output_padding: Union[int, tuple] = (0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_2d_tuple(kernel),
            stride=_make_2d_tuple(stride),
            dilation=_make_2d_tuple(dilation),
            padding=_make_2d_tuple(padding),
            padding_mode=padding_mode,
            output_channels=output_chan,
            output_padding=_make_2d_tuple(output_padding),
            bias=bias,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        padding_mode,
        bias,
        kernel,
        output_channels,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.ConvTranspose2d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

        return module

    @classmethod
    def predict_size(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        kernel,
        output_channels,
        output_padding,
        **kwargs,
    ) -> tuple:
        output_h = (
            (input_size[1] - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel[0] - 1)
            + output_padding[0]
            + 1
        )
        output_w = (
            (input_size[2] - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (kernel[1] - 1)
            + output_padding[1]
            + 1
        )

        return (output_channels, output_h, output_w)


@register_layer
class Deconv3d:
    TYPE = "deconv3d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: Union[int, str] = 1,
        stride: Union[int, tuple] = (1, 1, 1),
        dilation: Union[int, tuple] = (1, 1, 1),
        padding: Union[int, tuple] = (0, 0, 0),
        output_padding: Union[int, tuple] = (0, 0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=_make_3d_tuple(kernel),
            stride=_make_3d_tuple(stride),
            dilation=_make_3d_tuple(dilation),
            padding=_make_3d_tuple(padding),
            padding_mode=padding_mode,
            output_channels=output_chan,
            output_padding=_make_3d_tuple(output_padding),
            bias=bias,
        )

    @classmethod
    def make(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        padding_mode,
        bias,
        kernel,
        output_channels,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        module = torch.nn.ConvTranspose3d(
            in_channels=input_channel,
            out_channels=output_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

        return module

    @classmethod
    def predict_size(
        cls,
        input_size,
        stride,
        dilation,
        padding,
        kernel,
        output_channels,
        output_padding,
        **kwargs,
    ) -> tuple:
        output_d = (
            (input_size[1] - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel[0] - 1)
            + output_padding[0]
            + 1
        )

        output_h = (
            (input_size[2] - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (kernel[1] - 1)
            + output_padding[1]
            + 1
        )

        output_w = (
            (input_size[3] - 1) * stride[2]
            - 2 * padding[2]
            + dilation[2] * (kernel[2] - 1)
            + output_padding[2]
            + 1
        )

        return (output_channels, output_d, output_h, output_w)


@register_layer
class Repeat:
    TYPE = "repeat"

    @classmethod
    def elem(cls, sequence: list, iterations: int) -> dict:
        assert int(iterations)
        assert sequence
        return dict(type=cls.TYPE, sequence=sequence, iterations=iterations)

    @classmethod
    def make(cls, input_size, sequence, iterations, context, **kwargs) -> Module:
        seq, output_size = build_topology_from_list(
            sequence=sequence,
            input_size=input_size,
            context=context,
        )

        return layer.Repeat(seq, iterations)

    @classmethod
    def predict_size(cls, input_size, sequence, **kwargs) -> tuple:
        return predict_size(sequence=sequence, input_size=input_size)


@register_layer
class ConvModule:
    TYPE = "conv_module"

    @classmethod
    def elem(cls, module: list, dim: int) -> dict:
        assert isinstance(dim, int)
        assert module
        return dict(type=cls.TYPE, module=module, dim=dim)

    @staticmethod
    def shape_rm_dim(base: tuple, dim: int):
        out = list(base)
        out.pop(dim)
        return tuple(out)

    @staticmethod
    def shape_add_dim(base: tuple, dim: int, val: int):
        if isinstance(base, int):
            base = [base]
        out = list(base)
        out.insert(dim, val)
        return tuple(out)

    @classmethod
    def make(cls, input_size, module, dim, context, **kwargs) -> Module:
        seq, _ = build_topology_from_list(
            sequence=module,
            input_size=ConvModule.shape_rm_dim(input_size, dim),
            context=context,
        )

        return layer.ConvModule(seq, dim + 1)

    @classmethod
    def predict_size(cls, input_size, module, dim, **kwargs) -> tuple:
        iterations = input_size[dim]
        out = predict_size(sequence=module, input_size=ConvModule.shape_rm_dim(input_size, dim))
        return ConvModule.shape_add_dim(out, dim, iterations)


@register_layer
class SubpixelConv2d:
    TYPE = "subpixelconv2d"

    @classmethod
    def elem(
        cls,
        kernel: Union[int, tuple],
        output_chan: int = 3,
        scale_factor: int = 2,
        stride: tuple = (1, 1),
        dilation: tuple = (1, 1),
        padding: tuple = (0, 0),
        padding_mode: str = "zeros",
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            output_channels=output_chan,
            kernel=_make_2d_tuple(kernel),
            stride=_make_2d_tuple(stride),
            dilation=_make_2d_tuple(dilation),
            padding=_make_2d_tuple(padding),
            padding_mode=padding_mode,
            bias=bias,
            scale_factor=scale_factor,
        )

    @classmethod
    def make(
        cls,
        input_size,
        scale_factor,
        stride,
        dilation,
        padding,
        padding_mode,
        bias,
        kernel,
        output_channels,
        **kwargs,
    ) -> Module:
        input_channel = input_size[0]

        return layer.SubPixelConv2d(
            in_channels=input_channel,
            out_channels=output_channels,
            scale_factor=scale_factor,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

    @classmethod
    def predict_size(cls, input_size, scale_factor, output_channels, **kwargs) -> tuple:
        return (output_channels, input_size[1] * scale_factor, input_size[2] * scale_factor)


@register_layer
class SubblockConv2d:
    TYPE = "subblockconv2d"

    @classmethod
    def elem(
        cls,
        output_chan: int = 3,
        scale_factor: int = 2,
        bias: bool = True,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            output_channels=output_chan,
            bias=bias,
            scale_factor=scale_factor,
        )

    @classmethod
    def make(cls, input_size, scale_factor, bias, output_channels, **kwargs) -> Module:
        input_channels = input_size[0]

        return layer.SubBlockConv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            scale_factor=scale_factor,
            bias=bias,
        )

    @classmethod
    def predict_size(cls, input_size, scale_factor, output_channels, **kwargs) -> tuple:
        return (output_channels, input_size[1] * scale_factor, input_size[2] * scale_factor)


@register_layer
class AutoFlow:
    TYPE = "autoflow"

    @classmethod
    def elem(cls, nb_data: int, output_shape: torch.Size) -> dict:
        return dict(type=cls.TYPE, nb_data=nb_data, output_size=output_shape)

    @classmethod
    def make(cls, nb_data, output_size, **kwargs) -> Module:
        return layer.AutoFlow(nb_data, output_size)

    @classmethod
    def predict_size(cls, input_size, output_size, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return output_size

        mult = 1
        for i in range(0, len(input_size) - 1):
            mult *= input_size[i]

        return (mult, *_to_tuple(output_size))


@register_layer
class Concat:
    TYPE = "concat"

    @classmethod
    def elem(cls, sequences: list, dim: int = 0) -> dict:
        if not isinstance(sequences, list):
            sequences = [sequences]

        for i in range(len(sequences)):
            sub_sequence = sequences[i]
            if not isinstance(sub_sequence, list):
                sequences[i] = [sub_sequence]

        return dict(type=cls.TYPE, sequences=sequences, dim=dim)

    @classmethod
    def make(cls, input_size, sequences, dim, context, **kwargs) -> Module:

        assert sequences, "sequences has to be a non empty list"
        assert dim >= 0, "dim has to be positive or None"

        actual_dim = dim + 1

        built_sequences = []

        for sequence in sequences:
            built_sequence, output_size = build_topology_from_list(
                sequence=sequence,
                input_size=input_size,
                context=context,
            )
            built_sequences.append(built_sequence)

        # output shape is supposed to be the same for each sub sequence
        return layer.Concat(built_sequences, actual_dim)

    @classmethod
    def predict_size(cls, input_size, sequences, dim, **kwargs) -> tuple:

        assert sequences, "sequences has to be a non empty list"
        assert dim >= 0, "dim has to be positive or None"

        dim_cat = 0
        for sequence in sequences:
            output_size = predict_size(sequence=sequence, input_size=input_size)
            dim_cat += output_size[dim]

        output_size = list(output_size)
        output_size[dim] = dim_cat
        output_size = tuple(output_size)

        return output_size


@register_layer
class Sum:
    TYPE = "sum"

    @classmethod
    def elem(cls, dim: Optional[int] = None) -> dict:
        return dict(type=cls.TYPE, dim=dim)

    @classmethod
    def make(cls, dim, **kwargs) -> Module:
        dim = None if dim is None else dim + 1
        return layer.Sum(dim=dim)

    @classmethod
    def predict_size(cls, input_size, sequences, **kwargs) -> tuple:
        return input_size


@register_layer
class Add:
    TYPE = "add"

    @classmethod
    def elem(cls, sequences: List[list]) -> dict:
        return dict(type=cls.TYPE, sequences=sequences)

    @classmethod
    def make(cls, input_size, sequences, context, **kwargs) -> Module:
        assert sequences, "sequences has to be a non empty list"

        built_sequences = []
        for sequence in sequences:
            built_sequence, output_size = build_topology_from_list(
                sequence=sequence,
                input_size=input_size,
                context=context,
            )
            built_sequences.append(built_sequence)

        # output shape is supposed to be the same for each sub sequence
        return layer.Add(built_sequences)

    @classmethod
    def predict_size(cls, input_size, sequences, **kwargs) -> tuple:
        assert sequences, "sequences has to be a non empty list"
        return predict_size(sequence=sequences[0], input_size=input_size)


@register_layer
class Mul:
    TYPE = "mul"

    @classmethod
    def elem(cls, sequences: List[list]) -> dict:
        return dict(type=cls.TYPE, sequences=sequences)

    @classmethod
    def make(cls, input_size, sequences, context, **kwargs) -> Module:
        assert sequences, "sequences has to be a non empty list"

        built_sequences = []
        for sequence in sequences:
            built_sequence, output_size = build_topology_from_list(
                sequence=sequence,
                input_size=input_size,
                context=context,
            )
            built_sequences.append(built_sequence)

        # output shape is supposed to be the same for each sub sequence
        return layer.Mul(built_sequences)

    @classmethod
    def predict_size(cls, input_size, sequences, **kwargs) -> tuple:
        assert sequences, "sequences has to be a non empty list"
        return predict_size(sequence=sequences[0], input_size=input_size)


@register_layer
class IndexSelect:
    TYPE = "index_select"

    @classmethod
    def elem(cls, dim: int, index: List[int]) -> dict:
        return dict(type=cls.TYPE, dim=dim, index=index)

    @classmethod
    def make(cls, dim, index, **kwargs) -> Module:
        return layer.IndexSelect(dim=dim + 1, index=index)

    @classmethod
    def predict_size(cls, input_size, dim, index, **kwargs) -> tuple:
        output_size = list(copy.deepcopy(input_size))
        output_size[dim] = len(index)
        return tuple(output_size)


@register_layer
class InstanceNorm:
    TYPE = "instancenorm"

    @classmethod
    def elem(cls, eps: float = 1e-5, momentum: float = 0.1) -> dict:
        return dict(type=cls.TYPE, epsilon=eps, momentum=momentum)

    @classmethod
    def make(cls, input_size, epsilon, momentum, **kwargs) -> Module:
        return torch.nn.InstanceNorm2d(input_size[0], eps=epsilon, momentum=momentum)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class GroupNorm:
    TYPE = "group_norm"

    @classmethod
    def elem(cls, num_groups: int, num_channels: int) -> dict:
        return dict(type=cls.TYPE, num_groups=num_groups, num_channels=num_channels)

    @classmethod
    def make(cls, input_size, num_groups, num_channels, **kwargs) -> Module:
        return torch.nn.GroupNorm(num_groups, num_channels)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class TimeSlider:
    TYPE = "time_slider"

    @classmethod
    def elem(cls, window: int) -> dict:
        return dict(type=cls.TYPE, window=window)

    @classmethod
    def make(cls, input_size, window, **kwargs) -> Module:
        return layer.TimeSlider(window)

    @classmethod
    def predict_size(cls, input_size, window, **kwargs) -> tuple:
        if not isinstance(input_size, int) and len(input_size) != 1:
            raise RuntimeError("TimeSlider: Bad input size, normally 1.")
        return (window, input_size)


@register_layer
class Permute:
    TYPE = "permute"

    @classmethod
    def elem(cls, *dims) -> dict:
        return dict(type=cls.TYPE, dims=dims)

    @classmethod
    def make(cls, input_size, dims, **kwargs) -> Module:
        return layer.PermuteModule(*dims)

    @classmethod
    def predict_size(cls, input_size, dims, **kwargs) -> tuple:
        output_size = []
        for dim in dims:
            output_size.append(input_size[dim])
        return tuple(output_size)


@register_layer
class Switch:
    TYPE = "switch"

    @classmethod
    def elem(cls, sections: list, modules: list) -> dict:
        return dict(type=cls.TYPE, sections=sections, modules=modules)

    @classmethod
    def make(cls, input_size, sections, modules, context, **kwargs) -> Module:
        actual_modules = []

        for module in modules:
            module, _ = build_topology_from_list(
                sequence=module,
                input_size=input_size,
                context=context,
            )
            actual_modules.append(module)

        return layer.Switch(sections=sections, modules=actual_modules)

    @classmethod
    def predict_size(cls, input_size, modules, **kwargs) -> tuple:
        # All modules should have the same output size
        return predict_size(sequence=modules[0], input_size=input_size)


@register_layer
class SwitchIndexed:
    TYPE = "switch_indexed"

    @classmethod
    def elem(cls, index_file: list, nb_frames: int, modules: list) -> dict:
        return dict(type=cls.TYPE, file=index_file, nb_frames=nb_frames, modules=modules)

    @classmethod
    def make(cls, input_size, file, nb_frames, modules, context, **kwargs) -> Module:
        actual_modules = []

        for module in modules:
            module, _ = build_topology_from_list(
                sequence=module,
                input_size=input_size,
                context=context,
            )
            actual_modules.append(module)

        return layer.SwitchIndexed(nb_frames=nb_frames, index_file=file, modules=actual_modules)

    @classmethod
    def predict_size(cls, input_size, modules, **kwargs) -> tuple:
        # All modules should have the same output size
        return predict_size(sequence=modules[0], input_size=input_size)


@register_layer
class HsvToRgb:
    TYPE = "hsv_to_rgb"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return kornia.color.hsv.HsvToRgb()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class PixelUnshuffle:
    TYPE = "pixel_unshuffle"

    @classmethod
    def elem(cls, downscale_factor: int) -> dict:
        return dict(type=cls.TYPE, downscale_factor=downscale_factor)

    @classmethod
    def make(cls, input_size, downscale_factor, **kwargs) -> Module:
        return torch.nn.PixelUnshuffle(downscale_factor)

    @classmethod
    def predict_size(cls, input_size, downscale_factor, **kwargs) -> tuple:
        c_in, h_in, w_in = input_size[-3:]

        ouput_size = tuple(
            [
                *input_size[:-3],
                c_in * downscale_factor * downscale_factor,
                int(h_in / downscale_factor),
                int(w_in / downscale_factor),
            ]
        )
        return ouput_size


@register_layer
class YuvToRgb:
    TYPE = "yuv_to_rgb"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return kornia.color.yuv.YuvToRgb()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


@register_layer
class Crop2d:
    TYPE = "crop2d"

    @classmethod
    def elem(cls, th: int, tw: int) -> dict:
        return dict(type=cls.TYPE, th=th, tw=tw)

    @classmethod
    def make(cls, th: int, tw: int, **kwargs) -> Module:
        return layer.Crop2d(th, tw)

    @classmethod
    def predict_size(cls, input_size, th: int, tw: int, **kwargs) -> tuple:
        assert len(input_size) >= 3
        assert input_size[-1] >= tw
        assert input_size[-2] >= th

        return (*input_size[:-2], th, tw)


@register_layer
class WeightNorm:
    TYPE = "weightnorm"

    @classmethod
    def elem(cls, sequence: list) -> dict:
        return dict(type=cls.TYPE, sequence=sequence)

    @classmethod
    def make(cls, input_size, sequence: list, context, **kwargs) -> Module:
        seq, _ = build_topology_from_list(
            sequence=sequence,
            input_size=input_size,
            context=context,
        )

        return layer.WeightNorm(seq)

    @classmethod
    def predict_size(cls, input_size, sequence: list, **kwargs) -> tuple:
        return predict_size(sequence, input_size)


EyeInput = register_layer(layer.BuilderEyeInput)
GrayCodeInput = register_layer(layer.BuilderGrayCodeInput)
BinaryInput = register_layer(layer.BuilderBinaryInput)
HybridInput = register_layer(layer.BuilderHybridInput)
BinarySinusInput = register_layer(layer.BuilderBinarySinusInput)
IncrementalGrayCodeInput = register_layer(layer.BuilderIncrementalGrayCodeInput)
BinaryTriangleInput = register_layer(layer.BuilderBinaryTriangleInput)
BinaryToothInput = register_layer(layer.BuilderBinaryToothInput)
RegularizedGrayCodeInput = register_layer(layer.BuilderRegularizedGrayCodeInput)
Discretize = register_layer(layer.BuilderDiscretize)
DiscretizedPReLU = register_layer(layer.BuilderDiscretizedPReLU)
Yuv420ToYuv444 = register_layer(layer.BuilderYuv420ToYuv444)
Yuv420astride = register_layer(layer.BuilderYuv420astride)
FourierFeature = register_layer(layer.BuilderFourierFeature)
Noise = register_layer(layer.BuilderNoise)
PReLUB = register_layer(layer.BuilderPReLUB)
PixelShuffle = register_layer(layer.BuilderPixelShuffle)
ChannelUpscale = register_layer(layer.BuilderChannelUpscale)
Invol2d = register_layer(layer.BuilderInvolution2d)
ResBlock = register_layer(layer.BuilderResBlock)
AsymParamOutput = register_layer(layer.BuilderAsymParamOutput)
ContextSave = register_layer(layer.BuilderContextSave)
DevLayer = register_layer(layer.BuilderDevLayer)
XeLU = register_layer(layer.BuilderXeLU)
HardPish = register_layer(layer.BuilderHardPish)


QuantizedFourierFeature = register_layer(qlayer.BuilderQuantizedFourierFeature)
QuantizedHardPish = register_layer(qlayer.BuilderQuantizedHardPish)


def _method_exists(instance, methodname):
    return hasattr(instance, methodname) and callable(getattr(instance, methodname))


def __retrocompat_layer_desc(layer_class, desc_layer) -> dict:
    """
    Retrocompatibility adjustments: If the method argcompat exist, use it to transform the original
    arguments into updated ones.
    """
    if _method_exists(layer_class, "argcompat"):
        return layer_class.argcompat(desc_layer)
    return desc_layer


def predict_size(sequence: list, input_size: tuple) -> Tuple[int]:
    if not sequence:
        return input_size

    if isinstance(sequence, dict):
        sequence = sequence["sequence"]

    previous_size = input_size

    for desc_layer in sequence:
        layer_class = layers_factory()[desc_layer["type"]]
        desc_layer = __retrocompat_layer_desc(layer_class, desc_layer)
        previous_size = layer_class.predict_size(input_size=previous_size, **desc_layer)

    return previous_size


def build_topology_from_list(
    sequence: list, input_size: tuple = (1,), context: dict = None
) -> Tuple[Sequential, tuple]:
    if not sequence:
        return None

    if isinstance(sequence, dict):
        sequence = sequence["sequence"]

    context = dict() if context is None else context

    modules = []
    previous_size = input_size

    for desc_layer in sequence:
        layer_class = layers_factory()[desc_layer["type"]]
        desc_layer = __retrocompat_layer_desc(layer_class, desc_layer)
        output_size = layer_class.predict_size(input_size=previous_size, **desc_layer)
        module = layer_class.make(input_size=previous_size, context=context, **desc_layer)

        previous_size = output_size

        modules.append(module)

    return Sequential(*modules), previous_size
