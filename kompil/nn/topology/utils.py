import copy
import torch
from typing import List, Dict, Iterable, Tuple
from torch.nn import Module

import kompil.nn.topology.builder as builder


__COMPOSITE_LAYERS = [builder.ResBlock.TYPE]


def get_composite_layers():
    return copy.deepcopy(__COMPOSITE_LAYERS)


def is_composite(layer_type: str):
    return layer_type in __COMPOSITE_LAYERS


def module_to_topology_iterable(module: Module, topology: Dict) -> Iterable[Tuple[Module, dict]]:
    """Recursively find all the topology nodes and its matching module."""

    # For a Sequential or a ModuleList case, the topology only defines it as a list, let's
    # therefore move directly to their children.
    if isinstance(topology, list):
        assert isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList))

        for i in range(len(module)):
            iterator = module_to_topology_iterable(module[i], topology[i])
            for child_mod, child_topo in iterator:
                yield child_mod, child_topo

        return

    # A node is a dictionary with a defined type
    assert isinstance(topology, Dict), f"{topology}"

    typ = topology.get("type", None)
    this_is_composite = is_composite(typ)

    yield module, topology

    # Let's iterate over the module children to quantize the topology recursively
    for name, child in module.named_children():

        # FloatFunctional are not defined in the topology so we nee to skip the related modules
        # or list of modules.
        from torch.nn.quantized import FloatFunctional

        if isinstance(child, FloatFunctional):
            continue

        if all(
            isinstance(
                sub,
                (
                    FloatFunctional,
                    torch.nn.Identity,
                    torch.nn.ModuleList,
                    torch.nn.Sequential,
                ),
            )
            for sub in child.modules()
        ):
            continue

        # In the case of composites, children may not be in the topology and are handled by the
        # layer directly and its parameter "quantized", so we need to skip them. hopefully its the
        # right ones to be skipped.
        if this_is_composite and name not in topology:
            continue

        # Change child
        iterator = module_to_topology_iterable(child, topology[name])
        for child_mod, child_topo in iterator:
            yield child_mod, child_topo
