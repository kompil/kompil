import torch
from typing import List, Union
from kompil.utils.factory import Factory

__FACTORY = Factory("topology")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_topology(name: str):
    """
    Register the defined function as a topology list

    The function has to be as the following:
    fct(out_shape: torch.Size, nb_frames: int, model_extra: list) -> list
    """
    return factory().register(name)


def pattern_to_topology(
    pattern: str, out_shape: torch.Size, nb_frames: int, model_extra: list
) -> list:
    """
    Build a topology based on registered patterns
    """
    pattern_builder = factory()[pattern]

    return pattern_builder(
        out_shape=out_shape,
        nb_frames=nb_frames,
        model_extra=model_extra,
    )


def topology_from_model_file(file_path: str):
    data = torch.load(file_path, map_location="cpu")
    meta = data["model_meta_dict"]
    return meta["topology"]
