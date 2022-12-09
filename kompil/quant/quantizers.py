import copy
import torch
import abc

from typing import List, Dict
from torch.nn.intrinsic import _FusedModule
from torch.ao.quantization.stubs import QuantStub, DeQuantStub
from torch.nn import Module
from torch.quantization.quantization_mappings import get_default_static_quant_module_mappings
from torch.quantization import (
    swap_module,
    get_default_qconfig_propagation_list,
    is_activation_post_process,
    add_observer_,
)
from kompil.nn.layers.quantized.dequantize import DeQuantize
from kompil.nn.topology.utils import module_to_topology_iterable, is_composite
from kompil.nn.models.model import VideoNet
from kompil.data.timeline import create_timeline
from kompil.quant.quantizable import existing_quant_areas
from kompil.quant.factory import register_quantization
from kompil.quant.mapping import (
    get_f2o_mapping,
    get_o2q_mapping,
    get_qconf_mapping,
    get_f2q_topology_mapping,
)
from kompil.quant.utils.norm import normalize_model

__QUANTIZABLE_TAG = "quantizable"


def _is_quantizable(module: Module) -> bool:
    return hasattr(module, __QUANTIZABLE_TAG) and module.quantizable


def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output"""
    return self.activation_post_process(output)


def _apply_qconfig(module: Module, qconfig: dict) -> VideoNet:
    if type(module) in qconfig:
        module.qconfig = qconfig[type(module)]
    else:
        module.qconfig = qconfig["default"]

    return module


def _remove_activation_post_process(module: Module):
    if hasattr(module, "activation_post_process") and is_activation_post_process(
        module.activation_post_process
    ):
        delattr(module, "activation_post_process")

    # Remove activation_post_proceess hook
    handle_ids_to_remove = set()

    for handle_id, hook_fn in module._forward_hooks.items():
        if hook_fn is _observer_forward_hook:
            handle_ids_to_remove.add(handle_id)

    for handle_id in handle_ids_to_remove:
        module._forward_hooks.pop(handle_id)


def _remove_qconfig(module: Module):
    for child in module.children():
        _remove_qconfig(child)

    if hasattr(module, "qconfig"):
        del module.qconfig

    _remove_activation_post_process(module)


def _update_topology(module: Module, topology: Dict, mapping: Dict) -> List:
    topology = copy.deepcopy(topology)

    for child_mod, child_topo in module_to_topology_iterable(module, topology):
        # If the module is to be quantized, the topology has to be modified accordingly.
        typ = child_topo.get("type", None)
        if _is_quantizable(child_mod) and typ:
            if typ in mapping:
                child_topo["type"] = mapping[typ]
            elif is_composite(typ):
                child_topo["quantized"] = True

    return topology


def _untag_module(module: Module):
    for child in module.children():
        _untag_module(child)

    if hasattr(module, __QUANTIZABLE_TAG):
        delattr(module, __QUANTIZABLE_TAG)


def _tag_module(module: Module, in_quant_area: bool = False):
    setattr(module, __QUANTIZABLE_TAG, in_quant_area)

    for child in module.children():
        if isinstance(child, QuantStub):
            assert not in_quant_area
            in_quant_area = True
            setattr(child, __QUANTIZABLE_TAG, True)
        elif isinstance(child, DeQuantStub):
            assert in_quant_area
            setattr(child, __QUANTIZABLE_TAG, True)
            in_quant_area = False
        else:
            _tag_module(child, in_quant_area)


def _propagate_qconfig(
    module, qconfig_dict, qconfig, allow_list=None, qconfig_parent=None, prefix=""
):
    r"""
    Copied from pytorch "_propagate_qconfig_helper" in order to replace where to add qconfig.

    This is a helper function for `propagate_qconfig_`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        allow_list: list of quantizable modules
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict

    Return:
        None, module is modified inplace with qconfig attached
    """
    # TODO: Add test
    if allow_list is None:
        allow_list = get_default_qconfig_propagation_list()

    module_qconfig = qconfig_dict.get(type(module), qconfig_parent)
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)
    module_qconfig = getattr(module, "qconfig", module_qconfig)

    torch.quantization.qconfig.assert_valid_qconfig(module_qconfig, module)

    # Custom modif: add only for quantizable modules
    if _is_quantizable(module):
        _apply_qconfig(module, qconfig)
        qconfig_with_device_check = module.qconfig
    else:
        qconfig_with_device_check = None

    for name, child in module.named_children():
        module_prefix = prefix + "." + name if prefix else name
        _propagate_qconfig(
            child, qconfig_dict, qconfig, allow_list, qconfig_with_device_check, module_prefix
        )


def _prepare_model(model: VideoNet, c_config: dict, qconfig: dict) -> VideoNet:
    qconfig_propagation_list = get_default_qconfig_propagation_list()

    _propagate_qconfig(model, {}, qconfig=qconfig)

    add_observer_(
        model,
        qconfig_propagation_list=qconfig_propagation_list,
        custom_module_class_mapping=c_config,
    )

    return model


def _convert_model(model: VideoNet, c_config: dict) -> VideoNet:
    # Must be done on CPU
    model.cpu()
    mapping = get_default_static_quant_module_mappings()

    def _rec(module: Module):
        reassign = {}

        for name, submodule in module.named_children():
            if not isinstance(submodule, _FusedModule):
                _rec(submodule)

            # Dequantstub hasn't qconfig, handle it manualy
            if isinstance(submodule, DeQuantStub):
                reassign[name] = DeQuantize()
                continue

            reassign[name] = swap_module(submodule, mapping, c_config)

        for key, value in reassign.items():
            module._modules[key] = value

        return module

    _rec(model)

    return model


def _common_quantize(model: VideoNet, backend: str, qconfig: dict, forward: callable) -> VideoNet:
    # Setup the backend
    torch.backends.quantized.engine = backend

    assert existing_quant_areas(
        model
    ), "Provided model doesn't contain any quant area; see make_quantizable CLI"

    # Tag quantizable layer
    _tag_module(model)

    # Update only layer with 'quantizable' attribute
    # Not inplace
    model.topology = _update_topology(model.sequence, model.topology, get_f2q_topology_mapping())

    # Apply the appropriate qconfig to each layer into a quant area
    # Each layer with qconfig will be supplid by an observer object
    _prepare_model(model, get_f2o_mapping(), qconfig)

    # Compute scale and zero point according to observation
    # The attached observer will record min/max value from layer output
    forward(model)

    # Final convertion to quantized equivalent parametrized according previous observations
    _convert_model(model, get_o2q_mapping())

    # Remove qconfig attribute
    _remove_qconfig(model)

    # Untag quantizable layer
    _untag_module(model)

    return model


class _BaseQuantizer(abc.ABC):
    @abc.abstractmethod
    def quantize(self, model: VideoNet, blacklist: dict) -> VideoNet:
        pass

    @property
    @abc.abstractmethod
    def version(self) -> str:
        pass


@register_quantization("cypher")
class CypherQuantizer(_BaseQuantizer):
    __VERSION = 0

    def __init__(self):
        pass

    @property
    def version(self) -> str:
        return self.__VERSION

    def quantize(self, model: VideoNet) -> VideoNet:
        backend = "qnnpack"
        qconfig = get_qconf_mapping()["cypher"]

        model = copy.deepcopy(model).eval()

        def calibrate(model: VideoNet):
            device = torch.device("cuda")
            nb_frames = model.nb_frames
            timeline = create_timeline(nb_frames, device=device)

            model = model.to(device)

            for frame_id in range(nb_frames):
                time_vec = timeline[frame_id]
                x = time_vec.unsqueeze(0)
                _ = model(x)[0]

        qmodel = _common_quantize(model, backend, qconfig, calibrate)

        return qmodel


@register_quantization("dozer")
class DozerQuantizer(_BaseQuantizer):
    __VERSION = 0

    def __init__(self):
        pass

    @property
    def version(self) -> str:
        return self.__VERSION

    def quantize(self, model: VideoNet) -> VideoNet:
        nmodel = normalize_model(model)
        return self.__quantize(nmodel)

    def __quantize(self, model: VideoNet) -> VideoNet:
        backend = "qnnpack"
        qconfig = get_qconf_mapping()["cypher"]

        model = copy.deepcopy(model).eval()

        def calibrate(model: VideoNet):
            device = torch.device("cuda")
            nb_frames = model.nb_frames
            timeline = create_timeline(nb_frames, device=device)

            model = model.to(device)

            for frame_id in range(nb_frames):
                time_vec = timeline[frame_id]
                x = time_vec.unsqueeze(0)
                _ = model(x)[0]

        qmodel = _common_quantize(model, backend, qconfig, calibrate)

        return qmodel
