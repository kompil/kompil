import math
import torch
import numpy as np

import kompil.nn.topology.topology as topo

from kompil.nn.topology.pattern.registration import register_topology
from typing import List

__BASE = [32, 64, 128, 256, 256, 292, 580, 292, 256, 128, 128, 64]
__BASE_PARAMS = 8.24 * 1e6


def decoder_donnager_320p(
    deconv1: int,
    deconv2: int,
    deconv3: int,
    deconv4: int,
    deconv5: int,
    deconv6: int,
    deconv7: int,
    deconv8: int,
    deconv9: int,
    discretize: bool = True,
) -> list:
    """
    Decoder part of donnager.
    """
    base = [
        topo.deconv2d(kernel=2, output_chan=deconv1, stride=2, padding=(0, 0)),
        topo.prelu(),  # 2 x 2
        topo.deconv2d(kernel=2, output_chan=deconv2, stride=2, padding=(0, 0)),
        topo.prelu(),  # 4 x 4
        topo.deconv2d(kernel=2, output_chan=deconv3, stride=2, padding=(1, 0)),
        topo.prelu(),  # 6 x 8
        topo.deconv2d(kernel=4, output_chan=deconv4, stride=2, padding=(2, 1)),
        topo.prelu(),  # 10 x 16
        topo.deconv2d(kernel=4, output_chan=deconv5, stride=2, padding=(1, 2)),
        topo.prelu(),  # 20 x 30
        topo.deconv2d(kernel=4, output_chan=deconv6, stride=2, padding=1),
        topo.prelu(),  # 40 x 60
        topo.deconv2d(kernel=4, output_chan=deconv7, stride=2, padding=1),
        topo.prelu(),  # 80 x 120
        topo.deconv2d(kernel=4, output_chan=deconv8, stride=2, padding=1),
        topo.prelu(),  # 160 x 240
        topo.deconv2d(kernel=4, output_chan=deconv9, stride=2),
        topo.prelu(),  # 320 x 480
        topo.conv2d(kernel=3, output_chan=3),
    ]
    if discretize:
        base.extend([topo.prelu(), topo.discretize()])
    return base


def pattern_donnager_320p(
    nb_frames: int,
    linears: List[int],
    deconv1: int,
    deconv2: int,
    deconv3: int,
    deconv4: int,
    deconv5: int,
    deconv6: int,
    deconv7: int,
    deconv8: int,
    deconv9: int,
    discretize: bool = True,
) -> list:
    """
    Full donnager configuration
    """
    linear_topo = []
    for lin in linears:
        linear_topo.append(topo.linear(output_size=lin))
        linear_topo.append(topo.prelu())

    return [
        topo.in_graycode(nb_frames),
        *linear_topo,
        topo.reshape(shape=(linears[-1], 1, 1)),
        *decoder_donnager_320p(
            deconv1,
            deconv2,
            deconv3,
            deconv4,
            deconv5,
            deconv6,
            deconv7,
            deconv8,
            deconv9,
            discretize,
        ),
    ]


@register_topology("donnager")
def donnager(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert len(model_extra) > 10
    linear_values = [int(val) for val in model_extra[:-9]]
    deconv_values = [int(val) for val in model_extra[-9:]]

    return pattern_donnager_320p(nb_frames, linear_values, *deconv_values)


@register_topology("donnager_flow")
def donnager_flow(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert len(model_extra) == 10
    flow_size = int(model_extra[0])
    deconv_values = [int(val) for val in model_extra[1:]]

    return [
        topo.autoflow(nb_data=nb_frames, output_shape=flow_size),
        topo.reshape((flow_size, 1, 1)),
        *decoder_donnager_320p(*deconv_values),
    ]


@register_topology("donnager_ones")
def donnager_ones(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert not model_extra

    return [
        topo.autoflow(nb_data=nb_frames, output_shape=1),
        topo.reshape((1, 1, 1)),
        *decoder_donnager_320p(1, 1, 1, 1, 1, 1, 1, 1, 1, True),
    ]


@register_topology("donnager_mp")
def donnager_mp(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert len(model_extra) == 1
    p_count = float(model_extra[0]) * 1e6

    # scale up to requested mp
    donnager_extra = np.array([int(val * math.sqrt(p_count / __BASE_PARAMS)) for val in __BASE])

    # multiple of 4 are quite faster for many gpu friendly operations
    modulo = 4
    donnager_extra -= donnager_extra % modulo
    # just in case, clamp to [mod, inf[ to avoid '0' channels
    donnager_extra = np.clip(donnager_extra, modulo, math.inf)

    return donnager(out_shape, nb_frames, donnager_extra)


@register_topology("donnager_section_dynamic")
def donnager_section_dynamic(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert (
        model_extra is not None and len(model_extra) == 2
    ), "Model extra must contains 2 values : (nb_frames_in_whole_video, nb_million_params_in_whole_video)"

    total_video_nb_frames = int(model_extra[0])
    total_video_nb_mparam = float(model_extra[1])  # value in mp

    nb_mparam = (nb_frames / total_video_nb_frames) * total_video_nb_mparam

    # scale up to requested mp
    p_count = nb_mparam * 1e6
    donnager_extra = np.array([int(val * math.sqrt(p_count / __BASE_PARAMS)) for val in __BASE])

    # multiple of 4 are quite faster for many gpu friendly operations
    modulo = 4
    donnager_extra -= donnager_extra % modulo
    # just in case, clamp to [mod, inf[ to avoid '0' channels
    donnager_extra = np.clip(donnager_extra, modulo, math.inf)

    return donnager(out_shape, nb_frames, donnager_extra)


@register_topology("donnager_section_walloc")
def donnager_section_walloc(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert (
        model_extra is not None and len(model_extra) == 2
    ), "Model extra must contains 2 values : (path/to/walloc.pth, idx)"

    walloc_fpath = str(model_extra[0])
    sect_idx = int(model_extra[1])

    nb_param = torch.load(walloc_fpath)[sect_idx]

    # scale up to requested mp
    donnager_extra = np.array([int(val * math.sqrt(nb_param / __BASE_PARAMS)) for val in __BASE])

    # multiple of 4 are quite faster for many gpu friendly operations
    modulo = 4
    donnager_extra -= donnager_extra % modulo
    # just in case, clamp to [mod, inf[ to avoid '0' channels
    donnager_extra = np.clip(donnager_extra, modulo, math.inf)

    return donnager(out_shape, nb_frames, donnager_extra)


def decoder_donnager_720p(
    deconv1: int,
    deconv2: int,
    deconv3: int,
    deconv4: int,
    deconv5: int,
    deconv6: int,
    deconv7: int,
    deconv8: int,
    deconv9: int,
    deconv10: int,
    discretize: bool = True,
) -> list:
    """
    Decoder part of donnager 720p.
    """
    base = [
        topo.deconv2d(kernel=2, output_chan=deconv1, stride=2, padding=(0, 0)),
        topo.mprelu(),
        topo.deconv2d(kernel=2, output_chan=deconv2, stride=2, padding=(0, 0)),
        topo.mprelu(),
        topo.deconv2d(kernel=2, output_chan=deconv3, stride=2, padding=(0, 0)),
        topo.mprelu(),
        topo.deconv2d(kernel=2, output_chan=deconv4, stride=2, padding=(0, 0)),
        topo.mprelu(),
        topo.deconv2d(kernel=(2, 4), output_chan=deconv4, stride=1, padding=(3, 0)),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=deconv5, stride=2, padding=(1, 0)),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=deconv6, stride=2, padding=(1, 1)),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=deconv7, stride=2, padding=(0, 1)),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=deconv8, stride=2, padding=(1, 1)),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=deconv9, stride=2, padding=(1, 1)),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=deconv10, stride=2, padding=(0, 0)),
        topo.mprelu(),
        topo.conv2d(kernel=3, output_chan=3),
    ]
    if discretize:
        base.extend([topo.prelu(), topo.discretize()])
    return base


def pattern_donnager_720p(
    nb_frames: int,
    linears: List[int],
    deconv1: int,
    deconv2: int,
    deconv3: int,
    deconv4: int,
    deconv5: int,
    deconv6: int,
    deconv7: int,
    deconv8: int,
    deconv9: int,
    deconv10: int,
    discretize: bool = True,
):
    linear_topo = []
    for lin in linears:
        linear_topo.append(topo.linear(output_size=lin))
        linear_topo.append(topo.prelu())

    return [
        topo.in_graycode(nb_frames),
        *linear_topo,
        topo.reshape(shape=(linears[-1], 1, 1)),
        *decoder_donnager_720p(
            deconv1,
            deconv2,
            deconv3,
            deconv4,
            deconv5,
            deconv6,
            deconv7,
            deconv8,
            deconv9,
            deconv10,
            discretize,
        ),
    ]


@register_topology("donnager_720p")
def donnager_720p(out_shape: torch.Size, nb_frames: int, model_extra: list):
    """
    avatar 20s tested with [64, 128, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
    """
    assert len(model_extra) > 11
    linear_values = [int(val) for val in model_extra[:-10]]
    deconv_values = [int(val) for val in model_extra[-10:]]

    return pattern_donnager_720p(nb_frames, linear_values, *deconv_values)


__BASE_720 = [32, 64, 100, 128, 160, 200, 300, 200, 160, 128, 64, 32, 24, 16]
__BASE_720_PARAMS = 2.06 * 1e6


@register_topology("donnager_720p_mp")
def donnager_720p_mp(out_shape: torch.Size, nb_frames: int, model_extra: list):
    assert len(model_extra) == 1
    p_count = float(model_extra[0]) * 1e6

    # scale up to requested mp
    donnager_extra = np.array(
        [int(val * math.sqrt(p_count / __BASE_720_PARAMS)) for val in __BASE_720]
    )

    # multiple of 4 are quite faster for many gpu friendly operations
    modulo = 4
    donnager_extra -= donnager_extra % modulo
    # just in case, clamp to [mod, inf[ to avoid '0' channels
    donnager_extra = np.clip(donnager_extra, modulo, math.inf)

    return donnager_720p(out_shape, nb_frames, donnager_extra)
