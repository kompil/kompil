import math
import copy
import torch
import random
import numpy as np
from typing import List

import kompil.nn.topology.topology as topo

from kompil.nn.topology.pattern.registration import register_topology

__BASE = [12, 28, 60, 124, 124, 140, 284, 140, 124, 60, 60, 28]
__BASE_PARAMS = 1.99 * 1e6


def daedalus_decoder(*channels: List[int]) -> list:
    """Decoder part of Daedalus"""
    # Tools
    _channels = copy.deepcopy(list(*channels))

    def pop_chan():
        return _channels.pop(0)

    # Constants
    f = 2
    f2 = f * f
    # Build topology
    first_c = pop_chan()
    return [
        topo.linear(output_size=first_c * f2),
        topo.prelu(),
        topo.reshape(shape=(first_c, f, f)),
        topo.conv2d(kernel=1, output_chan=pop_chan() * f2),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=(2, 1), output_chan=pop_chan() * f2),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2, padding=(1, 1)),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2, padding=(0, 1)),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=2, output_chan=pop_chan() * f2),
        topo.prelu(),
        topo.conv2d(kernel=4, output_chan=6, padding=1),
        topo.prelu(),
    ]


def daedalus_pattern(nb_frames: int, *channels: List[int]) -> list:
    """
    Full donnager configuration
    """
    linear_topo = []
    for c in channels[:-9]:
        linear_topo.extend([topo.linear(output_size=c), topo.prelu()])

    return [
        topo.in_incremental_graycode(nb_frames),
        *linear_topo,
        *daedalus_decoder(channels[-9:]),
    ]


@register_topology("daedalus_yuv420")
def daedalus(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert len(model_extra) > 10
    channels = [int(value) for value in model_extra]
    return daedalus_pattern(nb_frames, *channels)


@register_topology("daedalus_yuv420_mp")
def daedalus_mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
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
    return daedalus_pattern(nb_frames, *channels)
