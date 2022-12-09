import torch
import abc
import copy

from typing import List, Dict
from torch.nn.modules.container import Sequential

from torch.nn import Module
from torch.nn.quantized import FloatFunctional, QFunctional

from kompil.nn.models.model import VideoNet
from kompil.nn.layers.corrected.base import _Corrected
from kompil.data.timeline import create_timeline
from kompil.corr.factory import register_corrector
from kompil.corr.mapping import get_correction_mapping, get_q2c_mapping, get_q2c_topology_mapping
from kompil.nn.topology.utils import module_to_topology_iterable


def _swap_module(
    module: Module,
    output_error: torch.Tensor,
    corrected_mapping: Dict,
    correction_mapping: Dict,
    context: Dict,
) -> Module:
    new_module = module
    module_cls = type(module)

    if module_cls in corrected_mapping:
        if module_cls in correction_mapping:
            correction_cls = correction_mapping[module_cls]
        else:
            correction_cls = correction_mapping["default"]

        correction = correction_cls.from_data(module, output_error)
        new_module = corrected_mapping[module_cls].from_quantized(module, correction, context)
        new_module.to(output_error.device)

    return new_module


def _get_module_outputs(module: Module, model: VideoNet, forward: callable):
    outputs = None

    def _hook_output(hooked_output: torch.Tensor):
        nonlocal outputs
        outputs = hooked_output if outputs is None else torch.cat([outputs, hooked_output], dim=0)

    handle = module.register_forward_hook(lambda l, _, outp: _hook_output(outp))

    forward(model)

    handle.remove()

    return outputs


def _observe_outputs_error(omodule: Module, qmodule: Module, omodel: VideoNet, qmodel: VideoNet):
    def _forward(model: VideoNet):
        nb_frames = model.nb_frames
        timeline = create_timeline(nb_frames, device=model.device)

        for frame_id in range(nb_frames):
            time_vec = timeline[frame_id]
            x = time_vec.unsqueeze(0)

            with torch.no_grad():
                _ = model(x)[0]

    qoutputs = _get_module_outputs(qmodule, qmodel, _forward)
    ooutputs = _get_module_outputs(omodule, omodel, _forward).to(qoutputs.device)

    with torch.no_grad():
        errors = ooutputs - qoutputs.dequantize()
        print("Error average (L1) :", torch.abs(torch.mean(errors)).item())

    return errors


def _update_topology(module: Module, topology: Dict, mapping: Dict) -> List:
    topology = copy.deepcopy(topology)

    for child_mod, child_topo in module_to_topology_iterable(module, topology):
        # If the module is to be quantized, the topology has to be modified accordingly.
        type = child_topo.get("type", None)

        # Module must be a corrected one
        if type and type in mapping and isinstance(child_mod, _Corrected):
            child_topo["type"] = mapping[type]

    return topology


class _BaseCorrector(abc.ABC):
    @abc.abstractmethod
    def correct(self, omodel: VideoNet, qmodel: VideoNet) -> VideoNet:
        pass

    @property
    @abc.abstractmethod
    def version(self) -> str:
        pass


@register_corrector("harkonnen")
class HarkonnenCorrector(_BaseCorrector):
    __VERSION = 0

    @property
    def version(self) -> str:
        return self.__VERSION

    def correct(self, omodel: VideoNet, qmodel: VideoNet) -> VideoNet:
        q2c_mapping = get_q2c_mapping()
        correction_mapping = get_correction_mapping()["harkonnen"]

        omodel = omodel.eval().cpu()
        cmodel = qmodel.eval().cpu()

        def _rec_correct(osequence: Sequential, qsequence: Sequential) -> Sequential:
            for name, qmodule in qsequence.named_children():
                # 1) Remove FloatFunctional and QFunctional because they are Observed/Quantized specific (in Kompil pipeline).
                # If we keep them, it can occurs some difference during the topo-module hierarchy browsing.
                # 2) Also, there is quite nothing to "correct" in these modules
                if isinstance(qmodule, (FloatFunctional, QFunctional)):
                    continue

                try:
                    omodule = osequence.get_submodule(name)
                except Exception as e:
                    print(
                        "You possibly forgot to pass the quantizable model as original input. Please make see make_quantizable."
                    )
                    raise e

                # If this module must be corrected and has a corrected equivalence
                if type(qmodule) in q2c_mapping and type(qmodule) in correction_mapping:
                    # Gather all the output (frame per frame) error : o - dq(q)
                    with torch.no_grad():
                        errors = _observe_outputs_error(omodule, qmodule, omodel, cmodel)
                    # Generate a corrected equivalent based on gathered error
                    # Need to add inference context because we need to run the just-corrected module for further corrections
                    qsequence._modules[name] = _swap_module(
                        qmodule, errors, q2c_mapping, correction_mapping, cmodel.inference_context
                    )
                # This correction does not rec on just-corrected module
                # Change it if needed
                elif len(list(omodule.children())) > 0:
                    qsequence._modules[name] = _rec_correct(omodule, qmodule)

            return qsequence

        cmodel.sequence = _rec_correct(omodel.sequence, cmodel.sequence)

        # Will update only "_Corrected" modules
        cmodel.topology = _update_topology(
            cmodel.sequence, cmodel.topology, get_q2c_topology_mapping()
        )

        return cmodel


@register_corrector("no_correction")
class NoCorrector(_BaseCorrector):
    __VERSION = 0

    @property
    def version(self) -> str:
        return self.__VERSION

    def correct(self, omodel: VideoNet, qmodel: VideoNet) -> VideoNet:
        return qmodel
