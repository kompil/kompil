import math
import torch
from kompil.nn.layers import prelub

import kompil.nn.topology.topology as topo
from kompil.nn.topology.pattern.registration import register_topology


@register_topology("example_cluster")
def example_cluster(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    out_shape = out_shape[-2:]

    return [
        topo.save(
            path="build/layers/autoflow_data.pth",
            sequence=[
                topo.autoflow(nb_data=nb_frames, output_shape=(1, 30, 30)),
            ],
        ),
        topo.adjacent2d(kernel=7, output_chan=1, output_size=(178, 320)),
        topo.prelu(),
        topo.adjacent2d(kernel=5, output_chan=1, output_size=(178, 320)),
        topo.prelu(),
        topo.adjacent2d(kernel=5, output_chan=3, output_size=out_shape),
        topo.prelu(),
    ]


@register_topology("example_save")
def example_save(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.linear(10),
        topo.prelu(),
        topo.save(
            path="build/layers/test_block_1.pth",
            sequence=[
                topo.linear(10),
                topo.prelu(),
            ],
        ),
        topo.save(
            path="build/layers/test_block_2.pth",
            sequence=[
                topo.linear(6 * 160 * 240),
                topo.prelu(),
                topo.reshape((6, 160, 240)),
            ],
        ),
    ]


@register_topology("example_load")
def example_load(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.linear(10),
        topo.prelu(),
        topo.load(
            path="build/layers/test_block_1.pth",
            learnable=True,
            sequence=[
                topo.linear(10),
                topo.prelu(),
            ],
        ),
        topo.load(
            path="build/layers/test_block_2.pth",
            learnable=False,
            sequence=[
                topo.linear(6 * 160 * 240),
                topo.prelu(),
                topo.reshape((6, 160, 240)),
            ],
        ),
    ]


@register_topology("example_subblock")
def example_subblock(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.autoflow(nb_data=nb_frames, output_shape=(128, 4, 4)),
        topo.deconv2d(kernel=2, output_chan=512, stride=2, padding=(1, 0)),
        topo.prelu(),  # 6 x 8
        topo.deconv2d(kernel=4, output_chan=512, stride=2, padding=(2, 1)),
        topo.prelu(),  # 10 x 16
        topo.deconv2d(kernel=4, output_chan=256, stride=2, padding=(1, 2)),
        topo.prelu(),  # 20 x 30
        topo.deconv2d(kernel=4, output_chan=256, stride=2, padding=1),
        topo.prelu(),  # 40 x 60
        topo.subblockconv2d(scale_factor=8, output_chan=3),
        topo.prelu(),
        topo.conv2d(kernel=3, output_chan=3, padding=1),
    ]


@register_topology("example_conv3d")
def example_conv3d(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    return [
        topo.in_graycode(nb_frames),
        topo.linear(output_size=75),
        topo.prelu(),
        topo.linear(output_size=50),
        topo.prelu(),
        topo.linear(output_size=100),
        topo.prelu(),
        topo.linear(output_size=c * 3 * h * w),
        topo.prelu(),
        topo.reshape(shape=(c, 3, h, w)),
        topo.conv3d(output_chan=3, kernel=3, padding=1),
        topo.prelu(),
        topo.conv3d(output_chan=2, kernel=3, padding=1),
        topo.prelu(),
        topo.reshape(shape=(6, h, w)),
    ]


@register_topology("example_adj1d")
def example_adj1d(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.in_graycode(nb_frames),
        topo.linear(output_size=75),
        topo.prelu(),
        topo.linear(output_size=100),
        topo.prelu(),
        topo.linear(output_size=1000),
        topo.prelu(),
        topo.reshape(shape=(1, 1000)),
        topo.adjacent1d(kernel=30, output_chan=1, output_size=2000),
        topo.prelu(),
        topo.adjacent1d(kernel=30, output_chan=1, output_size=10000),
        topo.prelu(),
        topo.adjacent1d(kernel=30, output_chan=1, output_size=10000),
        topo.prelu(),
        topo.adjacent1d(kernel=20, output_chan=1, output_size=10000),
        topo.prelu(),
        topo.adjacent1d(kernel=20, output_chan=1, output_size=100000),
        topo.prelu(),
        topo.adjacent1d(kernel=20, output_chan=1, output_size=100000),
        topo.prelu(),
        topo.adjacent1d(kernel=20, output_chan=1, output_size=100000),
        topo.prelu(),
        topo.adjacent1d(kernel=20, output_chan=1, output_size=100000),
        topo.prelu(),
        topo.adjacent1d(kernel=20, output_chan=1, output_size=100000),
        topo.prelu(),
        topo.adjacent1d(kernel=10, output_chan=3, output_size=320 * 480),
        topo.reshape(shape=(3, 320, 480)),
    ]


@register_topology("example_time_slider")
def example_time_slider(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    from kompil.nn.layers.inputs import get_gc_nodes

    return [
        topo.time_slider(window=3),
        topo.in_graycode(nb_frames),
        topo.reshape(shape=3 * get_gc_nodes(nb_frames)),
        topo.linear(output_size=c * h * w),
        topo.reshape(shape=out_shape),
    ]


@register_topology("example_time_slider_af")
def example_time_slider_mb(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    time_window = 3
    return [
        topo.time_slider(window=time_window),
        topo.autoflow(nb_data=nb_frames + time_window - 1, output_shape=(10, 10)),
        topo.reshape(shape=time_window * 10 * 10),
        topo.linear(output_size=c * h * w),
        topo.reshape(shape=out_shape),
    ]


@register_topology("example_permute")
def example_permute(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    return [
        topo.in_graycode(nb_frames),
        topo.linear(output_size=c * h * w),
        topo.reshape(shape=(h, c, w)),
        topo.permute(1, 0, 2),
    ]


@register_topology("example_model_extra")
def example_model_extra(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    assert len(model_extra) == 2
    return [
        topo.in_graycode(nb_frames),
        topo.linear(output_size=int(model_extra[0])),
        topo.prelu(),
        topo.linear(output_size=int(model_extra[1])),
        topo.prelu(),
        topo.linear(output_size=c * h * w),
        topo.prelu(),
        topo.reshape(shape=(c, h, w)),
    ]


@register_topology("example_autoflow")
def example_autoflow(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    return [
        topo.autoflow(nb_data=nb_frames, output_shape=100),
        topo.linear(output_size=c * h * w),
        topo.prelu(),
        topo.reshape(shape=(c, h, w)),
    ]


@register_topology("example_switch")
def example_switch(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    cut_at = int(nb_frames / 2)
    seq = [
        topo.in_graycode(nb_frames),
        topo.linear(output_size=100),
        topo.prelu(),
        topo.linear(output_size=100),
        topo.prelu(),
        topo.linear(output_size=c * h * w),
        topo.prelu(),
        topo.reshape(shape=(c, h, w)),
    ]
    import kompil.data.timeline as tl

    return [
        topo.switch(
            sections=[
                (0, 0, cut_at),
                (1, 0, nb_frames - cut_at),
            ],
            modules=[seq, seq],
        ),
    ]


@register_topology("example_switch_indexed")
def example_switch(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    cut_at = int(nb_frames / 2)
    seq = [
        topo.in_graycode(nb_frames),
        topo.linear(output_size=100),
        topo.prelu(),
        topo.linear(output_size=100),
        topo.prelu(),
        topo.linear(output_size=c * h * w),
        topo.prelu(),
        topo.reshape(shape=(c, h, w)),
    ]
    import kompil.data.timeline as tl

    return [
        topo.switch_indexed(
            index_file="build/clusters.pth",
            nb_frames=nb_frames,
            modules=[seq, seq],
        ),
    ]


@register_topology("example_yuv420")
def example_yuv420(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.in_graycode(nb_frames),
        topo.linear(output_size=32),
        topo.prelu(),
        topo.linear(output_size=6 * 160 * 240),
        topo.reshape(shape=(6, 160, 240)),
        topo.colorspace_420_to_444(),
        topo.yuv_to_rgb(),
        topo.prelu(),
        topo.discretize(),
    ]


@register_topology("example_crop2d")
def example_crop2d(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.in_graycode(nb_frames),
        topo.linear(output_size=16),
        topo.prelu(),
        topo.linear(output_size=3 * 340 * 500),
        topo.reshape(shape=(3, 340, 500)),
        topo.crop2d(320, 480),
    ]


@register_topology("example_concat")
def example_concat(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    oneseq = [
        topo.linear(320 * 480),
        topo.reshape((1, 320, 480)),
    ]
    return [
        topo.in_graycode(nb_frames),
        topo.concat(
            dim=0,
            sequences=[oneseq, oneseq, oneseq],
        ),
    ]


@register_topology("example_prune")
def example_prune(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.in_graycode(nb_frames),
        topo.prune(
            module=[
                topo.linear(10000),
                topo.prelu(),
            ]
        ),
        topo.linear(100),
        topo.linear(3),
        topo.prelu(),
        topo.prune(
            module=[
                topo.linear(math.prod(dim for dim in out_shape)),
                topo.reshape(out_shape),
            ]
        ),
    ]


@register_topology("example_conv_module")
def example_conv_module(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.in_graycode(nb_frames),
        topo.linear(2 * 10),
        topo.reshape((2, 10)),
        topo.conv_module(
            module=[
                topo.linear(3 * 160 * 240),
                topo.reshape((3, 160, 240)),
            ],
            dim=0,
        ),
        topo.reshape((6, 160, 240)),
    ]


@register_topology("example_resblock")
def example_resblock(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.linear(16 * 160 * 240),
        topo.reshape((16, 160, 240)),
        topo.resblock(
            [
                topo.conv2d(kernel=1, output_chan=16),
            ],
            select=6,
            dim=0,
        ),
        topo.resblock(
            [
                topo.conv2d(kernel=1, output_chan=6),
            ],
            select=6,
            dim=0,
        ),
        topo.resblock(
            [
                topo.conv2d(kernel=1, output_chan=6),
            ]
        ),
        topo.reshape((6, 160, 240)),
    ]


@register_topology("example_context_save")
def example_context_save(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [
        topo.context_save("input"),
        topo.in_graycode(nb_frames),
        topo.linear(output_size=10),
        topo.context_save("after_first_linear"),
        topo.prelu(),
        topo.linear(output_size=6 * 160 * 240),
        topo.reshape(shape=(6, 160, 240)),
        topo.prelu(),
    ]
