import os
import torch

import kompil.nn.topology.topology as topo

from typing import Tuple
from kompil.nn.topology.pattern.registration import (
    register_topology,
    topology_from_model_file,
    factory,
)
from kompil.nn.topology.pattern.donnager import pattern_donnager_320p, donnager_mp


CUTS = [
    0,
    142,
    706,
    2541,
    3374,
    5667,
    6800,
    7040,
    8705,
    10704,
    11581,
    13939,
    15638,
    18168,
    20735,
    24108,
    25298,
    29827,
    32206,
    34151,
    35248,
    36264,
    37951,
    39194,
    41396,
]


def section_data(sequence: int, nb_frames: int) -> Tuple[int, str]:
    assert sequence < len(CUTS)
    start = CUTS[sequence]
    end = CUTS[sequence + 1] - 1 if sequence < len(CUTS) - 1 else nb_frames - 1
    return end - start, f"silicon_valley-{start}-{end}.pth"


def define_with_donnager_mp(section: str, mp: int):
    def fct(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
        assert not model_extra

        return donnager_mp(out_shape, nb_frames, [mp])

    return factory().register_item(section, fct)


@register_topology("silicon_valley")
def silicon_valley(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert not model_extra
    import kompil.data.timeline as tl

    assert nb_frames == 43147

    switch_sections = []
    switch_modules = []
    for i, seq in enumerate(range(len(CUTS))):
        seq_duration, file_name = section_data(seq, nb_frames)
        file_path = os.path.join("build", "silicon_valley", file_name)
        seq_topo = topology_from_model_file(file_path)
        switch_sections.append((i, 0, seq_duration))
        switch_modules.append(
            [topo.load(sequence=seq_topo, path=file_path, learnable=True)],
        )

    return [topo.switch(sections=switch_sections, modules=switch_modules)]


@register_topology("sv_section_1")
@register_topology("sv_section_3")
@register_topology("sv_section_5")
@register_topology("sv_section_7")
@register_topology("sv_section_17")
@register_topology("sv_section_18")
@register_topology("sv_section_19")
@register_topology("sv_section_20")
@register_topology("sv_section_22")
@register_topology("sv_section_23")
@register_topology("sv_section_24")
def sv_model_1(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert not model_extra
    return pattern_donnager_320p(
        nb_frames, [100, 256, 384], 384, 384, 384, 384, 300, 256, 128, 64, 16
    )


@register_topology("sv_section_0")
@register_topology("sv_section_4")
@register_topology("sv_section_8")
@register_topology("sv_section_9")
@register_topology("sv_section_10")
@register_topology("sv_section_11")
@register_topology("sv_section_12")
@register_topology("sv_section_13")
@register_topology("sv_section_14")
@register_topology("sv_section_15")
@register_topology("sv_section_16")
@register_topology("sv_section_21")
def sv_model_2(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert not model_extra
    return pattern_donnager_320p(
        nb_frames, [100, 256, 512], 512, 512, 512, 512, 384, 256, 128, 64, 16
    )


@register_topology("sv_section_6")
def sv_section_6(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    assert not model_extra

    return [
        topo.autoflow(nb_data=nb_frames, output_shape=(512, 10, 15)),
        topo.prelu(),
        topo.deconv2d(kernel=4, output_chan=256, stride=2, padding=1),
        topo.prelu(),  # 20 x 30
        topo.deconv2d(kernel=4, output_chan=128, stride=2, padding=1),
        topo.prelu(),  # 40 x 60
        topo.deconv2d(kernel=4, output_chan=64, stride=2, padding=1),
        topo.prelu(),  # 80 x 120
        topo.deconv2d(kernel=4, output_chan=32, stride=2, padding=1),
        topo.prelu(),  # 160 x 240
        topo.deconv2d(kernel=4, output_chan=16, stride=2, padding=1),
        topo.prelu(),  # 320 x 480
        topo.deconv2d(kernel=3, output_chan=3, padding=1),
        # topo.hsv_to_rgb(),
        topo.prelu(),
        topo.discretize(),
    ]


define_with_donnager_mp("sv_section_2", 4)


@register_topology("sv_cluster")
def example_cluster(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    out_shape = out_shape[-2:]

    return [
        topo.save(
            path="build/layers/sv_cluster.pth",
            sequence=[
                topo.autoflow(nb_data=nb_frames, output_shape=(512, 1, 1)),
            ],
        ),
        topo.deconv2d(kernel=2, output_chan=512, stride=2, padding=(0, 0)),
        topo.prelu(),  # 2 x 2
        topo.deconv2d(kernel=2, output_chan=512, stride=2, padding=(0, 0)),
        topo.prelu(),  # 4 x 4
        topo.deconv2d(kernel=2, output_chan=256, stride=2, padding=(1, 0)),
        topo.prelu(),  # 6 x 8
        topo.deconv2d(kernel=4, output_chan=256, stride=2, padding=(2, 1)),
        topo.prelu(),  # 10 x 16
        topo.deconv2d(kernel=4, output_chan=128, stride=2, padding=(1, 2)),
        topo.prelu(),  # 20 x 30
        topo.deconv2d(kernel=4, output_chan=128, stride=2, padding=(1, 2)),
        topo.prelu(),  # 40 x 60
        topo.deconv2d(kernel=4, output_chan=64, stride=2, padding=1),
        topo.prelu(),  # 80 x 120
        topo.deconv2d(kernel=4, output_chan=32, stride=2, padding=1),
        topo.prelu(),  # 160 x 240
        topo.conv2d(kernel=3, output_chan=3),
        topo.upsample(out_shape),  # hard rescale
    ]
