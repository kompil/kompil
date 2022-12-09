import torch
import threading
from typing import List, Iterable

from torch.nn import Module
from torch.nn.utils.weight_norm import weight_norm, remove_weight_norm


class WeightNorm(Module):
    __constants__ = ["sequence"]

    def __init__(self, sequence):
        super().__init__()
        assert sequence
        self.sequence = sequence

    def weight_norm(self):
        def fn(m):
            if hasattr(m, "weight"):
                m = weight_norm(m)
            return m

        self.sequence = self.sequence.apply(fn)

    def forward(self, x):
        return self.sequence(x)

    def remove_weight_norm(self):
        def fn(m):
            if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
                m = remove_weight_norm(m)
            return m

        self.sequence = self.sequence.apply(fn)


def _recursive_iter_layertype(module: torch.nn.Module, ltype) -> Iterable[torch.nn.Module]:
    for mod in module.children():
        if isinstance(mod, ltype):
            yield mod
        for mod2 in _recursive_iter_layertype(mod, ltype):
            yield mod2


def norm_weight_norm_layers(module: torch.nn.Module):
    for mod in _recursive_iter_layertype(module, WeightNorm):
        mod.weight_norm()


def rm_norm_weight_norm_layers(module: torch.nn.Module):
    for mod in _recursive_iter_layertype(module, WeightNorm):
        mod.remove_weight_norm()
