import torch
import math
from typing import List, Optional, Dict

import kompil.nn.topology.topology as topo

from kompil.nn.topology.pattern.registration import register_topology


class _Activation:
    def __init__(self, std, end) -> None:
        self.std = std
        self.end = end


__ACTIVATIONS: Dict[str, _Activation] = {
    "prelu": _Activation(std=topo.prelu(), end=topo.prelu()),
    "mish": _Activation(std=topo.mish(), end=topo.mish()),
    "pish": _Activation(std=topo.pish(), end=topo.pish()),
    "wish": _Activation(std=topo.wish(), end=topo.wish()),
    "mixprelu": _Activation(std=topo.prelu(), end=topo.mprelu()),
    "asymp": _Activation(std=topo.prelu(), end=topo.out_asymp((0.5, 0.0, 0.0))),
}


def generic_aurora(
    input_module: dict,
    linear_channels: list,
    expand_channels: list,
    expand_pooling: int,
    tweak_pooling: int,
    tweak_relay: int,
    tweak_iteration: int,
    ff: Optional[int],
    activation: str,
    with_resblock: bool = True,
):
    # Args checks
    assert len(linear_channels) > 1
    assert len(expand_channels) == 9

    global __ACTIVATIONS
    activation = __ACTIVATIONS[activation]

    # Input conversion
    layers_input = [input_module]

    # Linear layers
    layers_linear = list()
    for chan_count in linear_channels:
        layers_linear.extend(
            [
                topo.linear(output_size=chan_count),
                activation.std,
            ]
        )

    # Expand layers
    layers_expand = [
        topo.linear(output_size=expand_channels[0] * 4),
        activation.std,
        topo.reshape(shape=(expand_channels[0], 2, 2)),
    ]

    if ff is not None:
        layers_expand.extend(
            [
                topo.pixelnorm(),
                topo.fourier_feature(ff),
            ]
        )

    layers_expand.extend(
        [
            topo.conv2d(kernel=1, output_chan=expand_channels[1] * 4),
            topo.pixel_shuffle(factor=2),
            activation.std,
            topo.conv2d(kernel=1, output_chan=expand_channels[2] * 4),
            topo.pixel_shuffle(factor=2),
            activation.std,
            topo.conv2d(kernel=1, output_chan=expand_channels[3] * 4),
            topo.pixel_shuffle(factor=2),
            topo.crop2d(10, 15),
            activation.std,
            topo.conv2d(kernel=1, output_chan=expand_pooling),
            topo.conv2d(kernel=3, output_chan=expand_channels[4] * 4, padding=1),
            topo.pixel_shuffle(factor=2),
            activation.std,
            topo.conv2d(kernel=3, output_chan=expand_channels[5] * 4, padding=1),
            topo.pixel_shuffle(factor=2),
            activation.std,
            topo.conv2d(kernel=3, output_chan=expand_channels[6] * 4, padding=1),
            topo.pixel_shuffle(factor=2),
            activation.std,
            topo.conv2d(kernel=3, output_chan=expand_channels[7] * 4, padding=1),
            activation.std,
            topo.pixel_shuffle(factor=2),
            topo.conv2d(kernel=3, output_chan=expand_channels[8], padding=1),
            activation.end,
        ]
    )

    if not with_resblock:

        layers_tweak = [
            topo.conv2d(kernel=3, output_chan=6, padding=1),
            activation.end,
        ]

    else:

        # Tweak layers
        one_tweak = [
            topo.resblock(
                [
                    topo.conv2d(kernel=1, output_chan=tweak_pooling),
                    activation.std,
                    topo.conv2d(kernel=3, output_chan=tweak_relay, padding=1),
                ],
                select=6,
                dim=0,
            ),
            activation.end,
        ]

        layers_tweak = []
        for _ in range(tweak_iteration):
            layers_tweak.extend(one_tweak)
        layers_tweak.extend(
            [
                topo.resblock(
                    [
                        topo.conv2d(kernel=1, output_chan=tweak_pooling),
                        activation.std,
                        topo.conv2d(kernel=3, output_chan=6, padding=1),
                    ],
                    select=6,
                    dim=0,
                ),
                activation.end,
            ]
        )

    # Put it together
    return [
        *layers_input,
        topo.quantize_stub(),
        *layers_linear,
        *layers_expand,
        *layers_tweak,
        topo.dequantize_stub(),
    ]


BASE_LINEAR_CHANNELS = [20, 48, 48, 48]
BASE_EXPAND_CHANNELS = [84, 92, 100, 200, 100, 84, 52, 40, 64]
BASE_EXPAND_POOLING = 68
BASE_TWEAK_POOLING = 16
BASE_TWEAK_RELAY = 32
BASE_TWEAK_ITERATION = 2
BASE_FF = None
BASE_SIZE = 1
BASE_ACTIVATION = "pish"


@register_topology("aurora_mp")
def aurora_mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:

    assert len(model_extra) == 1
    mp = float(model_extra[0])

    def _mult(val):
        return max(4, int((val * math.sqrt(mp / BASE_SIZE) // 4)) * 4)

    # Data
    linear_channels = [_mult(val) for val in BASE_LINEAR_CHANNELS]
    expand_channels = [_mult(val) for val in BASE_EXPAND_CHANNELS]
    expand_pooling = _mult(BASE_EXPAND_POOLING)
    tweak_pooling = _mult(BASE_TWEAK_POOLING)
    tweak_relay = _mult(BASE_TWEAK_RELAY)
    tweak_iteration = BASE_TWEAK_ITERATION
    ff = _mult(BASE_FF) if BASE_FF is not None else None

    return generic_aurora(
        input_module=topo.in_incremental_graycode(nb_frames),
        linear_channels=linear_channels,
        expand_channels=expand_channels,
        expand_pooling=expand_pooling,
        tweak_pooling=tweak_pooling,
        tweak_relay=tweak_relay,
        tweak_iteration=tweak_iteration,
        ff=ff,
        activation=BASE_ACTIVATION,
    )


@register_topology("aurora_ref")
def aurora_ref(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:

    return generic_aurora(
        input_module=topo.in_incremental_graycode(nb_frames),
        linear_channels=BASE_LINEAR_CHANNELS,
        expand_channels=BASE_EXPAND_CHANNELS,
        expand_pooling=BASE_EXPAND_POOLING,
        tweak_pooling=BASE_TWEAK_POOLING,
        tweak_relay=BASE_TWEAK_RELAY,
        tweak_iteration=BASE_TWEAK_ITERATION,
        ff=BASE_FF,
        activation=BASE_ACTIVATION,
    )


@register_topology("aurora_wish")
def aurora_wish(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:

    return generic_aurora(
        input_module=topo.in_incremental_graycode(nb_frames),
        linear_channels=BASE_LINEAR_CHANNELS,
        expand_channels=BASE_EXPAND_CHANNELS,
        expand_pooling=BASE_EXPAND_POOLING,
        tweak_pooling=BASE_TWEAK_POOLING,
        tweak_relay=BASE_TWEAK_RELAY,
        tweak_iteration=BASE_TWEAK_ITERATION,
        ff=BASE_FF,
        activation="wish",
    )


@register_topology("aurora_quant_ref")
def aurora_quant_ref(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:

    return generic_aurora(
        input_module=topo.in_incremental_graycode(nb_frames),
        linear_channels=BASE_LINEAR_CHANNELS,
        expand_channels=BASE_EXPAND_CHANNELS,
        expand_pooling=BASE_EXPAND_POOLING,
        tweak_pooling=BASE_TWEAK_POOLING,
        tweak_relay=BASE_TWEAK_RELAY,
        tweak_iteration=BASE_TWEAK_ITERATION,
        ff=BASE_FF,
        activation="prelu",
    )


@register_topology("aurora_quant_mp")
def aurora_quant_mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:

    assert len(model_extra) == 1
    mp = float(model_extra[0])

    def _mult(val):
        return max(4, int((val * math.sqrt(mp / BASE_SIZE) // 4)) * 4)

    # Data
    linear_channels = [_mult(val) for val in BASE_LINEAR_CHANNELS]
    expand_channels = [_mult(val) for val in BASE_EXPAND_CHANNELS]
    expand_pooling = _mult(BASE_EXPAND_POOLING)
    tweak_pooling = _mult(BASE_TWEAK_POOLING)
    tweak_relay = _mult(BASE_TWEAK_RELAY)
    tweak_iteration = BASE_TWEAK_ITERATION
    ff = _mult(BASE_FF) if BASE_FF is not None else None

    return generic_aurora(
        input_module=topo.in_incremental_graycode(nb_frames),
        linear_channels=linear_channels,
        expand_channels=expand_channels,
        expand_pooling=expand_pooling,
        tweak_pooling=tweak_pooling,
        tweak_relay=tweak_relay,
        tweak_iteration=tweak_iteration,
        ff=ff,
        activation="prelu",
        with_resblock=False,
    )


@register_topology("aurora_quant2_mp")
def aurora_quant_mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:

    assert len(model_extra) == 1
    mp = float(model_extra[0])

    def _mult(val):
        return max(4, int((val * math.sqrt(mp / BASE_SIZE) // 4)) * 4)

    # Data
    linear_channels = [_mult(val) for val in BASE_LINEAR_CHANNELS]
    expand_channels = [_mult(val) for val in BASE_EXPAND_CHANNELS]
    expand_pooling = _mult(BASE_EXPAND_POOLING)
    tweak_pooling = _mult(BASE_TWEAK_POOLING)
    tweak_relay = _mult(BASE_TWEAK_RELAY)
    tweak_iteration = BASE_TWEAK_ITERATION
    ff = _mult(BASE_FF) if BASE_FF is not None else None

    return generic_aurora(
        input_module=topo.in_regularized_graycode(nb_frames),
        linear_channels=linear_channels,
        expand_channels=expand_channels,
        expand_pooling=expand_pooling,
        tweak_pooling=tweak_pooling,
        tweak_relay=tweak_relay,
        tweak_iteration=tweak_iteration,
        ff=ff,
        activation="prelu",
        with_resblock=False,
    )


@register_topology("aurora_1920x880")
def aurora_1920x880(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert out_shape == (6, 440, 960)
    assert len(model_extra) == 1
    mp = float(model_extra[0])

    def _mult(val):
        return max(4, int((val * math.sqrt(mp / 2.06) // 4)) * 4)

    activation = topo.pish()

    linear_channels = [28, 64, 64, 64]
    expand_channels = [116, 116, 128, 140, 140, 116, 72, 72, 56, 48]
    expand_pooling = 96
    tweak_pooling = 16
    tweak_relay = 32

    linear_channels = [_mult(val) for val in linear_channels]
    expand_channels = [_mult(val) for val in expand_channels]
    expand_pooling = _mult(expand_pooling)
    tweak_pooling = _mult(tweak_pooling)
    tweak_relay = _mult(tweak_relay)

    return [
        topo.in_incremental_graycode(nb_frames),
        topo.linear(output_size=linear_channels[0]),
        activation,
        topo.linear(output_size=linear_channels[1]),
        activation,
        topo.linear(output_size=linear_channels[2]),
        activation,
        topo.linear(output_size=linear_channels[3]),
        activation,
        topo.linear(output_size=expand_channels[0] * 16),
        activation,
        topo.reshape(shape=(expand_channels[0], 4, 4)),
        topo.conv2d(kernel=1, output_chan=expand_channels[1] * 4),
        topo.pixel_shuffle(factor=2),
        activation,
        topo.conv2d(kernel=1, output_chan=expand_channels[2] * 4),
        topo.pixel_shuffle(factor=2),
        activation,
        topo.conv2d(kernel=1, output_chan=expand_channels[3] * 4),
        topo.pixel_shuffle(factor=2),
        topo.crop2d(20, 30),
        activation,
        topo.conv2d(kernel=1, output_chan=expand_pooling),
        topo.conv2d(kernel=3, output_chan=expand_channels[4] * 4, padding=1),
        topo.pixel_shuffle(factor=2),
        activation,
        topo.conv2d(kernel=3, output_chan=expand_channels[5] * 4, padding=1),
        topo.pixel_shuffle(factor=2),
        activation,
        topo.crop2d(70, 120),
        topo.conv2d(kernel=3, output_chan=expand_channels[6] * 4, padding=1),
        topo.pixel_shuffle(factor=2),
        topo.crop2d(110, 240),
        activation,
        topo.conv2d(kernel=3, output_chan=expand_channels[7] * 4, padding=1),
        topo.pixel_shuffle(factor=2),
        activation,
        topo.conv2d(kernel=3, output_chan=expand_channels[8] * 4, padding=1),
        topo.pixel_shuffle(factor=2),
        activation,
        topo.conv2d(kernel=3, output_chan=expand_channels[9], padding=1),
        activation,
        topo.resblock(
            [
                topo.conv2d(kernel=1, output_chan=tweak_pooling),
                activation,
                topo.conv2d(kernel=3, output_chan=tweak_relay, padding=1),
            ],
            select=6,
            dim=0,
        ),
        activation,
        topo.resblock(
            [
                topo.conv2d(kernel=1, output_chan=tweak_pooling),
                activation,
                topo.conv2d(kernel=3, output_chan=tweak_relay, padding=1),
            ],
            select=6,
            dim=0,
        ),
        activation,
        topo.resblock(
            [
                topo.conv2d(kernel=1, output_chan=tweak_pooling),
                activation,
                topo.conv2d(kernel=3, output_chan=6, padding=1),
            ],
            select=6,
            dim=0,
        ),
        activation,
    ]
