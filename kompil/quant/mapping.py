import copy

from kornia.color.yuv import YuvToRgb
from torch.nn import PReLU, Conv1d, Conv2d, Conv3d, Linear

from kompil.nn.layers.quantized import *
from kompil.nn.layers.pish import Pish
from kompil.nn.layers.pixelnorm import PixelNorm
from kompil.nn.layers.fourier import FourierFeature
from kompil.nn.layers.inputs import IncrementalGrayCodeInput, GrayCodeInput
from kompil.nn.layers.reshape import Reshape
from kompil.nn.topology import builder
from kompil.quant.qconfigs import (
    min_max_uint8_pt_qconfig,
    min_max_uint8_pta_int8_pcw_qconfig,
)


__FLOAT_TO_OBSERVED_MODULE_MAPPING = {
    Pish: ObservedPish,
    PixelNorm: ObservedPixelNorm,
    FourierFeature: ObservedFourierFeature,
    PReLU: ObservedPReLU,
    SPish: ObservedSPish,
    HardPish: ObservedHardPish,
}


__OBSERVED_TO_QUANT_MODULE_MAPPING = {
    ObservedPish: QuantizedPish,
    ObservedPixelNorm: QuantizedPixelNorm,
    ObservedFourierFeature: QuantizedFourierFeature,
    ObservedPReLU: QuantizedPReLU,
    ObservedSPish: QuantizedSPish,
    ObservedHardPish: QuantizedHardPish,
    YuvToRgb: QuantizedYuv444ToRgb,
}


__FLOAT_TO_QUANT_TOPOLOGY_MAPPING = {
    builder.Conv2d.TYPE: builder.QuantizedConv2d.TYPE,
    builder.PixelNorm.TYPE: builder.QuantizedPixelNorm.TYPE,
    builder.PReLU.TYPE: builder.QuantizedPReLU.TYPE,
    builder.Linear.TYPE: builder.QuantizedLinear.TYPE,
    builder.Pish.TYPE: builder.QuantizedPish.TYPE,
    builder.SPish.TYPE: builder.QuantizedSPish.TYPE,
    builder.FourierFeature.TYPE: builder.QuantizedFourierFeature.TYPE,
    builder.QuantizeStub.TYPE: builder.Quantize.TYPE,
    builder.FixedQuantizeStub.TYPE: builder.FixedQuantize.TYPE,
    builder.DeQuantizeStub.TYPE: builder.DeQuantize.TYPE,
    builder.HardPish.TYPE: builder.QuantizedHardPish.TYPE,
}

__QCONFIG_MAPPING = {
    "cypher": {
        "default": min_max_uint8_pt_qconfig,
        Conv1d: min_max_uint8_pta_int8_pcw_qconfig,
        Conv2d: min_max_uint8_pta_int8_pcw_qconfig,
        Conv3d: min_max_uint8_pta_int8_pcw_qconfig,
        Linear: min_max_uint8_pta_int8_pcw_qconfig,
    }
}

__BLACKLIST_MAPPING = {
    "inputs": {
        IncrementalGrayCodeInput,
        GrayCodeInput,
    },
    "1": {
        IncrementalGrayCodeInput,
        GrayCodeInput,
        Linear,
        PixelNorm,
        FourierFeature,
        Reshape,
        Pish,
    },
}


def get_f2o_mapping():
    return copy.deepcopy(__FLOAT_TO_OBSERVED_MODULE_MAPPING)


def get_o2q_mapping():
    return copy.deepcopy(__OBSERVED_TO_QUANT_MODULE_MAPPING)


def get_f2q_topology_mapping():
    return copy.deepcopy(__FLOAT_TO_QUANT_TOPOLOGY_MAPPING)


def get_qconf_mapping():
    return copy.deepcopy(__QCONFIG_MAPPING)


def get_blacklist_mapping():
    return copy.deepcopy(__BLACKLIST_MAPPING)
