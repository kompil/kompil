import torch

from torch.quantization.observer import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    RecordingObserver,
)

debug_observer = RecordingObserver


min_max_uint8_pt_observer = MinMaxObserver.with_args(
    reduce_range=False,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
)

min_max_int8_pt_observer = MinMaxObserver.with_args(
    reduce_range=False,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_affine,
)

min_max_int8_pc_observer = PerChannelMinMaxObserver.with_args(
    reduce_range=False,
    dtype=torch.qint8,
    qscheme=torch.per_channel_affine,
    ch_axis=0,
)

min_max_uint8_pcf_observer = PerChannelMinMaxObserver.with_args(
    reduce_range=False,
    dtype=torch.quint8,
    qscheme=torch.per_channel_affine_float_qparams,
    ch_axis=0,
)
