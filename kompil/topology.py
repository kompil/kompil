import os
import torch
from typing import Union, List, Any, Optional

from kompil.utils.numbers import to_scale
from kompil.nn.models.model import VideoNet, model_save
from kompil.nn.layers.save_load import load_load_layers
from kompil.nn.topology.pattern import pattern_to_topology
from kompil.nn.topology.builder import build_topology_from_list
from kompil.utils.video import resolution_to_chw
from kompil.utils.colorspace import convert_shape_to_colorspace
from kompil.nn.layers.prune import count_prunable_params


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def _output_shape(resolution: str, colorspace: str) -> torch.Size:
    c, h, w = resolution_to_chw(resolution)
    frame_shape = torch.Size((c, h, w))
    frame_shape = convert_shape_to_colorspace(frame_shape, "rgb8", colorspace)
    return frame_shape


def _to_topo_dict(
    topology_builder: str,
    nb_frames: float,
    colorspace: str,
    resolution: str,
    model_extra: Optional[List[Any]],
):
    frame_shape = _output_shape(resolution, colorspace)

    return pattern_to_topology(
        pattern=topology_builder,
        out_shape=frame_shape,
        nb_frames=nb_frames,
        model_extra=model_extra,
    )


def _to_model(nb_frames: float, resolution: Union[str, None], topo_dict: list):
    c, h, w = resolution_to_chw(resolution)
    model, output_shape = build_topology_from_list(topo_dict)
    return model, output_shape


def init(
    topology_builder: str,
    frames: int,
    colorspace: str,
    output: str,
    resolution: str,
    framerate: float,
    model_extra: Optional[List[Any]],
):
    # Get name
    filename = os.path.basename(output)
    model_name, ext = os.path.splitext(filename)

    # Generate model
    topo_dict = _to_topo_dict(topology_builder, frames, colorspace, resolution, model_extra)

    # Generate VideoNet
    model = VideoNet(
        name=model_name,
        nb_frames=frames,
        fps=framerate,
        frame_shape=_output_shape(resolution, colorspace),
        topology=topo_dict,
        colorspace=colorspace,
    )
    load_load_layers(model)

    # Save the model
    if os.path.exists(output):
        answer = input(f"File {output} exists. Do you want to erase it? [y/n] ")
        if answer.lower() not in ["yes", "y"]:
            print("Model not saved!")
            return

    model_save(model, output)
    print("Model saved!")


def show(
    topology_builder: str,
    frames: int,
    colorspace: str,
    resolution: str,
    framerate: float,
    model_extra: Union[List[Any], None],
):
    # Generate model
    topo_dict = _to_topo_dict(topology_builder, frames, colorspace, resolution, model_extra)
    model, output_shape = _to_model(frames, resolution, topo_dict)

    total_params = get_n_params(model)
    prune_params = count_prunable_params(model)
    prune_ratio = prune_params / total_params

    # Print infos
    print(model)
    print()
    print("Output shape:", output_shape)
    print("Num parameters:", to_scale(total_params))
    print("Prunable parameters:", to_scale(prune_params), f"({prune_ratio * 100:0.2f}%)")
    print()

    # Check output resolution
    c, h, w = _output_shape(resolution, colorspace)
    if tuple(output_shape) != (c, h, w):
        raise RuntimeError("Bad output shape")
