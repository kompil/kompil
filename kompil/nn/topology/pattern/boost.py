import math
import torch

from kompil.nn.topology.pattern.registration import register_topology
import kompil.nn.topology.topology as topo


@register_topology("boost-pre")
def pre_boost(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    """
    Test: 1220 epochs

    Min PSNR: 26.906875610351562
    Max PSNR: 43.71425247192383
    Mean PSNR: 32.578636169433594

    Min SSIM: 0.749754786491394
    Max SSIM: 0.9814361333847046
    Mean SSIM: 0.8963868021965027
    """
    return [
        topo.in_graycode(nb_frames),
        topo.save(
            uid="pre_boost",
            sequence=[
                topo.linear(output_size=50),
                topo.prelu(),
                topo.linear(output_size=100),
                topo.prelu(),
                topo.linear(output_size=100),
                topo.prelu(),
                topo.linear(output_size=1 * 80 * 120),
                topo.prelu(),
                topo.reshape(shape=(1, 80, 120)),
                topo.adjacent2d(kernel=6, output_chan=1, output_size=(320, 480)),
                topo.prelu(),
                topo.adjacent2d(kernel=6, output_chan=1, output_size=(320, 480)),
                topo.prelu(),
                topo.adjacent2d(kernel=6, output_chan=1, output_size=(320, 480)),
                topo.prelu(),
                topo.adjacent2d(kernel=6, output_chan=1, output_size=(320, 480)),
                topo.prelu(),
                topo.prelu(),
                topo.adjacent2d(kernel=6, output_chan=3, output_size=(320, 480)),
            ],
        ),
    ]


@register_topology("boost-post")
def post_boost(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.in_graycode(nb_frames),
        topo.load(
            uid="last_20s",
            learnable=False,
            sequence=[
                topo.linear(output_size=50),
                topo.prelu(),
                topo.linear(output_size=100),
                topo.prelu(),
                topo.linear(output_size=100),
                topo.prelu(),
                topo.linear(output_size=1 * 80 * 120),
                topo.prelu(),
                topo.reshape(shape=(1, 80, 120)),
                topo.adjacent2d(kernel=6, output_chan=1, output_size=(320, 480)),
                topo.prelu(),
                topo.adjacent2d(kernel=6, output_chan=1, output_size=(320, 480)),
                topo.prelu(),
                topo.adjacent2d(kernel=6, output_chan=1, output_size=(320, 480)),
                topo.prelu(),
                topo.adjacent2d(kernel=6, output_chan=1, output_size=(320, 480)),
                topo.prelu(),
                topo.prelu(),
                topo.adjacent2d(kernel=6, output_chan=3, output_size=(320, 480)),
            ],
        ),
        topo.prelu(),
        # topo.resblock(sequence=[
        # topo.adjacent2d(kernel=1, output_chan=1, output_size=(320 * 3, 480 * 3)),
        # topo.prelu(),
        # topo.adjacent2d(kernel=3, output_chan=1, output_size=(320 * 3, 480 * 3)),
        # topo.prelu(),
        # topo.adjacent2d(kernel=3, output_chan=1, output_size=(320 * 3, 480 * 3)),
        # topo.prelu(),
        # topo.adjacent2d(kernel=3, output_chan=1, output_size=(320 * 3, 480 * 3)),
        # topo.prelu(),
        # topo.adjacent2d(kernel=6, output_chan=3, output_size=(320, 480)),
        # ])
        topo.resblock(
            sequence=[
                topo.conv2d(
                    kernel=1,
                    output_chan=8,
                    padding_mode="replicate",
                    padding=1,
                    apply_weight_norm=True,
                ),
                topo.conv2d(
                    kernel=1,
                    output_chan=16,
                    apply_weight_norm=True,
                ),
                topo.prelu(),
                topo.conv2d(
                    kernel=3,
                    output_chan=3,
                    apply_weight_norm=True,
                ),
            ]
        ),
        topo.resblock(
            sequence=[
                topo.conv2d(
                    kernel=1,
                    output_chan=8,
                    padding_mode="replicate",
                    padding=1,
                    apply_weight_norm=True,
                ),
                topo.conv2d(
                    kernel=1,
                    output_chan=16,
                    apply_weight_norm=True,
                ),
                topo.prelu(),
                topo.conv2d(
                    kernel=3,
                    output_chan=3,
                    apply_weight_norm=True,
                ),
            ]
        ),
        topo.resblock(
            sequence=[
                topo.conv2d(
                    kernel=1,
                    output_chan=8,
                    padding_mode="replicate",
                    padding=1,
                    apply_weight_norm=True,
                ),
                topo.conv2d(
                    kernel=1,
                    output_chan=16,
                    apply_weight_norm=True,
                ),
                topo.prelu(),
                topo.conv2d(
                    kernel=3,
                    output_chan=3,
                    apply_weight_norm=True,
                ),
            ]
        ),
    ]
