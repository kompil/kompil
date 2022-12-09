import math
import torch

from kompil.nn.topology.pattern.registration import register_topology
import kompil.nn.topology.topology as topo


@register_topology("bigbang")
def topology_bigbang(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    c, h, w = out_shape
    h_div_4 = math.floor(h / 4)
    w_div_4 = math.floor(w / 4)

    topology = [topo.in_graycode(nb_frames)]
    log_nb_frames = math.log10(nb_frames)

    topology.append(topo.linear(output_size=5))
    topology.append(topo.mprelu())

    nb_linear = max(2, round(log_nb_frames * 0.75))
    for _ in range(0, nb_linear):
        topology.append(topo.linear(output_size=round(log_nb_frames * 100)))
        topology.append(topo.mprelu())

    topology.append(topo.linear(output_size=w_div_4 * h_div_4 * c))
    topology.append(topo.mprelu())
    topology.append(topo.reshape(shape=(c, h_div_4, w_div_4)))
    topology.append(topo.adjacent2d(kernel=7, output_chan=1, output_size=(h, w)))
    topology.append(topo.prelu())

    nb_duplication = max(1, int(log_nb_frames**2.2 - 7))
    one_seq = [
        topo.adjacent2d(
            kernel=5,
            output_chan=1,
            output_size=(h, w),
        ),
        topo.prelu(),
        topo.adjacent2d(
            kernel=5,
            output_chan=1,
            output_size=(h, w),
        ),
        topo.prelu(),
    ]
    topology.append(topo.add(sequences=[one_seq] * nb_duplication))

    topology.append(topo.adjacent2d(kernel=5, output_chan=c, output_size=(h, w)))
    topology.append(topo.prelu())
    topology.append(topo.discretize())

    return topology
