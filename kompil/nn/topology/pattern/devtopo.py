import torch

import kompil.nn.topology.topology as topo
from kompil.nn.topology.pattern.registration import register_topology


@register_topology("devtopo")
def devtopo(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    return [topo.devlayer(nb_frames=nb_frames, params=model_extra, outshape=out_shape)]
