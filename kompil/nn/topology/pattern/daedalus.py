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
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=3, output_chan=3),
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


@register_topology("daedalus")
def daedalus(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert len(model_extra) > 10
    channels = [int(value) for value in model_extra]
    return daedalus_pattern(nb_frames, *channels)


@register_topology("daedalus_2mp")
def daedalus_2mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return daedalus_pattern(nb_frames, *__BASE)


@register_topology("daedalus_mp")
def daedalus_mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert len(model_extra) == 1
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


def daedalus_complex_pattern(nb_frames: int, channels: List[int], kernels: List[int]) -> list:
    """Decoder part of Daedalus"""
    # Check kernels
    assert len(kernels) == 9
    for k in kernels:
        assert k > 0
    k0 = kernels[0]
    k1 = kernels[1]
    k2 = kernels[2]
    k3 = kernels[3]
    k4 = kernels[4]
    k5 = kernels[5]
    k6 = kernels[6]
    k7 = kernels[7]
    k8 = kernels[8]
    assert k0 == 1
    assert k1 < 3
    assert k2 > 1
    assert k8 % 2 == 1

    # Linears
    linear_topo = []
    for c in channels[:-9]:
        linear_topo.extend([topo.linear(output_size=c), topo.prelu()])

    # Tools for decoder
    _channels = copy.deepcopy(list(channels[-9:]))

    def pop_chan():
        return _channels.pop(0)

    # Constants
    f = 2
    f2 = f * f
    # Build topology
    first_c = pop_chan()
    return [
        topo.in_graycode(nb_frames),
        *linear_topo,
        topo.linear(output_size=first_c * f2),
        topo.prelu(),
        topo.reshape(shape=(first_c, f, f)),  # 2:2
        topo.conv2d(kernel=k0, output_chan=pop_chan() * f2),
        topo.pixel_shuffle(factor=2),  # 4:4
        topo.prelu(),
        topo.conv2d(kernel=(1 + k1, 0 + k1), output_chan=pop_chan() * f2, padding=k0 - 1),
        topo.pixel_shuffle(factor=2),  # 4:4
        topo.prelu(),
        topo.conv2d(kernel=k2, output_chan=pop_chan() * f2, padding=k1 - 1),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=k3, output_chan=pop_chan() * f2, padding=k2 - 2),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=k4, output_chan=pop_chan() * f2, padding=(k3 - 1, k3)),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=k5, output_chan=pop_chan() * f2, padding=k4 - 1),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=k6, output_chan=pop_chan() * f2, padding=k5 - 1),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=k7, output_chan=pop_chan() * f2, padding=k6),
        topo.pixel_shuffle(factor=2),
        topo.prelu(),
        topo.conv2d(kernel=k8, output_chan=3, padding=k7 - 1 + int((k8 - 5) / 2)),
        topo.prelu(),
        topo.discretize(),
    ]


@register_topology("daedalus_complex")
def daedalus_mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    channels = [int(v) for v in model_extra[:-9]]
    kernels = [int(v) for v in model_extra[-9:]]

    return daedalus_complex_pattern(nb_frames, channels, kernels)
