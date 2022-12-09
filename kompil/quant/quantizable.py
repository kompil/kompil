from typing import List, Dict, Tuple
from torch.nn import Sequential
from torch.nn.modules.module import Module
from torch.quantization import QuantStub, DeQuantStub

from kompil.nn.models.model import VideoNet
from kompil.nn.topology import topology as topo


def get_topo_children(layer: dict) -> list:
    if isinstance(layer, list):
        return layer

    subs = ["module", "modules", "sequence", "sequences"]

    for sub in subs:
        if sub in layer:
            return layer[sub]

    return None


def get_mod_children(module: Module):
    r"""
    This method returns from a module the list containing its consistant children.
    Consistant means meaningful for browsing the topo-module hierarchy.
    """
    from torch.nn.quantized import FloatFunctional, QFunctional

    # Get all current module children.
    children = list(module.children())

    #  Remove FloatFunctional and QFunctional because they are Observed/Quantized specific (in Kompil pipeline).
    # If we keep them, it can occurs some difference during the topo-module hierarchy browsing.
    matches = [x for x in children if x not in [FloatFunctional, QFunctional]]

    # In the case where the only children is a Sequential.
    # It can happen for Weighnorm or Resblock.
    # Basically, in the topology this sequential is not stored as a 'sequence' item, but directly in list instead.
    # So we return the Sequential to keep topo-module hierarchy on the same level.
    if len(matches) == 1 and isinstance(matches[0], Sequential):
        return matches[0]

    return matches


def _apply_automatic_quant_area(
    model: VideoNet, blacklist: list
) -> Tuple[List[Sequential], Sequential]:
    model_sequence = model.sequence
    model_topology = model.topology

    def _rec(sequence: Sequential, topology: List, in_quant_area) -> Tuple[Sequential, Dict, bool]:
        new_sequence = Sequential()
        new_topology = []

        new_idx = 0
        original_idx = 0

        children = get_mod_children(sequence)
        children_topology = get_topo_children(topology)

        # Browse every child to gather quantizable ones
        for i, submodule in enumerate(children):
            submodule_topology = children_topology[i]

            if get_topo_children(submodule_topology):
                submodule, submodule_topology, in_quant_area = _rec(
                    submodule,
                    submodule_topology,
                    in_quant_area,
                )
                new_sequence.add_module(str(new_idx), submodule)
                new_topology.append(submodule_topology)
                new_idx += 1

            elif type(submodule) not in blacklist:
                if not in_quant_area:
                    new_sequence.add_module(str(new_idx), QuantStub())
                    new_topology.append(topo.quantize_stub())
                    # We are in a whitelist area
                    in_quant_area = True
                    new_idx += 1

                new_sequence.add_module(str(new_idx), submodule)
                new_topology.append(submodule_topology)
                new_idx += 1

                # If it is the last module of the sequence, then close and wrap it
                if original_idx == len(children) - 1:
                    new_sequence.add_module(str(new_idx), DeQuantStub())
                    new_topology.append(topo.dequantize_stub())
                    new_idx += 1
                    in_quant_area = False

            # Considere blacklisted submodule
            else:
                #  Was in a whitelist area, then close and wrap it before doing anything if the current submodule
                if in_quant_area:
                    new_sequence.add_module(str(new_idx), DeQuantStub())
                    new_topology.append(topo.dequantize_stub())
                    new_idx += 1
                    in_quant_area = False

                # Add the blacklisted submodule as it
                new_sequence.add_module(str(new_idx), submodule)
                new_topology.append(submodule_topology)
                new_idx += 1

            original_idx += 1

        return new_sequence, topo.sequence(new_topology), in_quant_area

    new_sequence, new_topology, _ = _rec(model_sequence, model_topology, False)

    return new_sequence, new_topology


def existing_quant_areas(model: VideoNet) -> bool:
    ok = False

    def _rec(module: Module, in_quant_area):
        for child in list(module.children()):
            if isinstance(child, QuantStub):
                assert not in_quant_area
                in_quant_area = True
            if isinstance(child, DeQuantStub):
                assert in_quant_area
                in_quant_area = False
                nonlocal ok
                ok = True
            else:
                _rec(child, in_quant_area)

    _rec(model.sequence, False)

    return ok


def make_quantizable_model(model: VideoNet, blacklist: list) -> VideoNet:
    assert not existing_quant_areas(model)

    # Add surrounding Quant/DeQuant layer to not-blacklisted layers
    new_sequence, new_topology = _apply_automatic_quant_area(model, blacklist)

    # Inplace replacement of the sequence
    model.sequence = new_sequence
    model.topology = new_topology

    return model
