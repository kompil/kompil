import torch

import kompil.nn.topology.topology as topo
from kompil.nn.topology.pattern.registration import (
    register_topology,
    topology_from_model_file,
)


@register_topology("find_clusters")
def find_clusters(out_shape: torch.Size, nb_frames: int, model_extra: list):
    assert len(model_extra) == 10
    cfpath = model_extra[0]
    vals = [int(item) for item in model_extra[1:]]

    out_shape = out_shape[-2:]

    return [
        topo.save(
            path=cfpath,
            sequence=[
                topo.autoflow(nb_data=nb_frames, output_shape=vals[0]),
            ],
        ),
        topo.reshape((vals[0], 1, 1)),
        topo.deconv2d(kernel=2, output_chan=vals[1], stride=2, padding=(0, 0)),
        topo.prelu(),  # 2 x 2
        topo.deconv2d(kernel=2, output_chan=vals[2], stride=2, padding=(0, 0)),
        topo.prelu(),  # 4 x 4
        topo.deconv2d(kernel=2, output_chan=vals[3], stride=2, padding=(1, 0)),
        topo.prelu(),  # 6 x 8
        topo.deconv2d(kernel=4, output_chan=vals[4], stride=2, padding=(2, 1)),
        topo.prelu(),  # 10 x 16
        topo.deconv2d(kernel=4, output_chan=vals[5], stride=2, padding=(1, 2)),
        topo.prelu(),  # 20 x 30
        topo.deconv2d(kernel=4, output_chan=vals[6], stride=2, padding=(1, 2)),
        topo.prelu(),  # 40 x 60
        topo.deconv2d(kernel=4, output_chan=vals[7], stride=2, padding=1),
        topo.prelu(),  # 80 x 120
        topo.deconv2d(kernel=4, output_chan=vals[8], stride=2, padding=1),
        topo.prelu(),  # 160 x 240
        topo.conv2d(kernel=3, output_chan=3),
        topo.upsample(out_shape),  # hard rescale
        topo.prelu(),
        topo.discretize(),
    ]


@register_topology("find_clusters_simple")
def find_clusters_simple(out_shape: torch.Size, nb_frames: int, model_extra: list):
    assert len(model_extra) == 1
    cfpath = model_extra[0]

    out_shape = out_shape[-2:]

    return [
        topo.save(
            path=cfpath,
            sequence=[
                topo.autoflow(nb_data=nb_frames, output_shape=(1024)),
            ],
        ),
        topo.linear(1024),
        topo.prelu(),
        topo.reshape((64, 4, 4)),
        topo.deconv2d(kernel=2, output_chan=512, stride=2),
        topo.mprelu(),
        topo.deconv2d(kernel=2, output_chan=256, stride=2),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=128, stride=2),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=32, stride=2),
        topo.mprelu(),
        topo.deconv2d(kernel=4, output_chan=16, stride=2),
        topo.mprelu(),
        topo.conv2d(kernel=3, output_chan=8),
        topo.prelu(),
        topo.conv2d(kernel=3, output_chan=3),
        topo.upsample(out_shape),
        topo.prelu(),
        topo.discretize(),
    ]


@register_topology("auto_switch_cluster")
def auto_switch_cluster(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list:
    """
    :param model_extra: cluster_file_path, cluster_prefix
    """
    assert len(model_extra) == 2
    import kompil.data.timeline as tl

    cluster_prefix = str(model_extra[1])

    # Read clusters
    cluster_fpath = str(model_extra[0])
    t_clusters = torch.load(cluster_fpath)
    assert nb_frames == t_clusters.shape[0], "cluster and current video frames does not match"
    nb_clusters = t_clusters.max()

    # Build cluster_modules
    cluster_modules = []
    clust_histo = torch.bincount(t_clusters)
    for cluster_id, cluster_frames in enumerate(clust_histo):
        cluster_path = cluster_prefix + str(cluster_id) + ".pth"
        cluster_topology = topology_from_model_file(cluster_path)
        cluster_modules.append(
            [topo.load(sequence=cluster_topology, path=cluster_path, learnable=True)],
        )

    # Build switch
    return [
        topo.switch_indexed(index_file=cluster_fpath, nb_frames=nb_frames, modules=cluster_modules),
    ]
