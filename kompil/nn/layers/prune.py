import torch
from typing import Union, Tuple, Iterable
import torch.nn.utils.prune as prune
from torch.nn.modules.module import Module


class Prune(Module):

    __constants__ = ["__names", "__parameters_to_prune"]

    def __init__(self, module: Module, names=Union[str, Tuple[str]]):
        super().__init__()

        assert isinstance(module, torch.nn.Sequential)

        self.child = module
        self.__names = names if isinstance(names, list) or isinstance(names, tuple) else [names]

        self.__parameters_to_prune = []
        for module_elem in self.child.modules():
            for param_name in self.__names:
                if hasattr(module_elem, param_name):
                    self.__parameters_to_prune.append((module_elem, param_name))

        self.__start_params = None

    def forward(self, x):
        return self.child(x)

    def pruning_iter(self) -> Iterable[torch.Tensor]:
        for module, param_name in self.__parameters_to_prune:
            yield getattr(module, param_name).data

    def pruning_step(self, amount: float, pruning_method: prune.BasePruningMethod):
        prune.global_unstructured(
            self.__parameters_to_prune,
            pruning_method=pruning_method,
            amount=amount,
        )

    def pruning_end(self):
        for module, param_name in self.__parameters_to_prune:
            if hasattr(module, param_name + "_orig"):
                prune.remove(module, param_name)

    def pruning_save_params(self):
        self.__start_params = []
        for module, param_name in self.__parameters_to_prune:
            data = getattr(module, param_name).data.clone()
            self.__start_params.append((module, param_name, data))

    def pruning_reset_params(self):
        assert self.__start_params is not None, "parameters not saved"
        for module, param_name, data in self.__start_params:
            if hasattr(module, param_name + "_orig"):
                getattr(module, param_name + "_orig").data.copy_(data)
            else:
                getattr(module, param_name).data.copy_(data)


def recursive_iter_prune(module: torch.nn.Module) -> Iterable[Prune]:
    for mod in module.children():
        if isinstance(mod, Prune):
            yield mod
            # Alert user if there is nested pruning modules.
            for mod2 in recursive_iter_prune(mod):
                print("WARNING: pruning module ignored because it is nested in another one.")
        else:
            for mod2 in recursive_iter_prune(mod):
                yield mod2


def recursive_unprune(module: torch.nn.Module):
    for mod in recursive_iter_prune(module):
        mod.pruning_end()


def recursive_save_params(module: torch.nn.Module):
    for mod in recursive_iter_prune(module):
        mod.pruning_save_params()


def recursive_prune_step(
    module: torch.nn.Module, amount: float, ptype: prune.BasePruningMethod, lottery_ticket: bool
):
    for mod in recursive_iter_prune(module):
        mod.pruning_step(amount, ptype)
        if lottery_ticket:
            mod.pruning_reset_params()


def count_prunable_params(module: torch.nn.Module):
    return sum(t.numel() for m in recursive_iter_prune(module) for t in m.pruning_iter())
