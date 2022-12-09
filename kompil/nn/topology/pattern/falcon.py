import math
import copy
import torch
import numpy as np
from typing import List

import kompil.nn.topology.topology as topo

from kompil.nn.topology.pattern.registration import register_topology

__BASE = [25, 50, 80, 110, 124, 124, 140, 284, 140, 124, 60, 60, 28]
__BASE_PARAMS = 1.99 * 1e6


def falcon_decoder(*channels: List[int]) -> list:
    """Decoder part of Falcon"""
    # Tools
    _channels = copy.deepcopy(list(*channels))

    def pop_chan():
        return _channels.pop(0)

    # Constants
    f = 2
    f2 = f * f
    first_c = pop_chan()

    # Build topology
    return [
        topo.linear(output_size=first_c * f2),
        topo.pish(),
        topo.reshape(shape=(first_c, f, f)),
        topo.pixelnorm(),
        topo.fourier_feature(first_c),
        topo.conv2d(kernel=1, output_chan=pop_chan() * f2),
        topo.pish(),
        topo.pixel_shuffle(factor=2),
        topo.conv2d(kernel=(2, 1), output_chan=pop_chan() * f2),
        topo.pish(),
        topo.pixel_shuffle(factor=2),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2),
        topo.pish(),
        topo.pixel_shuffle(factor=2),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2, padding=(1, 1)),
        topo.pish(),
        topo.pixel_shuffle(factor=2),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2, padding=(0, 1)),
        topo.pish(),
        topo.pixel_shuffle(factor=2),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2),
        topo.pish(),
        topo.pixel_shuffle(factor=2),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2),
        topo.pish(),
        topo.pixel_shuffle(factor=2),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2),
        topo.pish(),
        topo.conv2d(kernel=3, output_chan=12, padding=1),
        topo.pish(),
        topo.conv2d(kernel=2, output_chan=6),
        topo.pish(),
    ]


def falcon_pattern(nb_frames: int, *channels: List[int]) -> list:
    """
    Full falcon configuration
    """
    linear_topo = []
    for c in channels[:-9]:
        linear_topo.extend([topo.linear(output_size=c), topo.pish()])

    return [
        topo.in_incremental_graycode(nb_frames),
        topo.weightnorm(
            sequence=[
                *linear_topo,
                *falcon_decoder(channels[-9:]),
            ]
        ),
    ]


@register_topology("falcon")
def falcon(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert len(model_extra) > 10
    channels = [int(value) for value in model_extra]
    return falcon_pattern(nb_frames, *channels)


@register_topology("falcon_mp")
def falcon_mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert len(model_extra) == 1
    assert out_shape[0] == 6
    # scale up to requested mp
    p_count = float(model_extra[0]) * 1e6
    ratio = math.sqrt(p_count / __BASE_PARAMS)
    channels = np.array([int(value * ratio) for value in __BASE])
    # multiple of 4 are quite faster for many gpu friendly operations
    modulo = 4
    channels -= channels % modulo
    # just in case, clamp to [mod, inf[ to avoid '0' channels
    channels = np.clip(channels, modulo, math.inf)
    # Create the topology
    channels = [int(value) for value in channels]
    return falcon_pattern(nb_frames, *channels)
