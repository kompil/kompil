import copy

from torch.nn.quantized import Conv2d

from kompil.nn.layers.corrected import *
from kompil.nn.layers.corrected.corrections import DummyCorrection

import kompil.nn.topology.builder as builder


__QUANTIZED_TO_CORRECTED_MODULE_MAPPING = {
    Conv2d: CorrectedConv2d,
}


__QUANTIZED_TO_CORRECTED_TOPOLOGY_MAPPING = {
    builder.QuantizedConv2d.TYPE: builder.CorrectedConv2d.TYPE,
}


__CORRECTION_MAPPING = {
    "harkonnen": {
        Conv2d: DummyCorrection,
    },
}


def get_q2c_mapping():
    return copy.deepcopy(__QUANTIZED_TO_CORRECTED_MODULE_MAPPING)


def get_q2c_topology_mapping():
    return copy.deepcopy(__QUANTIZED_TO_CORRECTED_TOPOLOGY_MAPPING)


def get_correction_mapping():
    return copy.deepcopy(__CORRECTION_MAPPING)
