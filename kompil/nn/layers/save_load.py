import torch
import os
from typing import List, Any

from jsonpatch import JsonPatch
from torch.nn.modules.module import Module
from kompil.utils.paths import make_dir
from kompil.utils.resources import get_pytorch_model


class SaveModule(Module):

    __constants__ = ["sequence", "sequence_list", "path"]

    def __init__(self, path, sequence, sequence_list):
        super().__init__()

        assert sequence
        assert path

        self.path = path
        self.sequence = sequence
        self.sequence_list = sequence_list

    def forward(self, x):
        return self.sequence(x)

    def save(self):
        make_dir(os.path.dirname(self.path))

        torch.save(
            {
                "version": 0,
                "model_state_dict": self.state_dict(),
                "model_meta_dict": {"topology": self.sequence_list},
            },
            self.path,
        )
        return self.path


class LoadModule(Module):

    __constants__ = ["sequence", "sequence_list", "path", "learnable"]

    def __init__(self, path, sequence, sequence_list, learnable):
        super().__init__()

        assert sequence
        assert path

        self.path = path
        self.sequence = sequence
        self.learnable = learnable
        self.sequence_list = sequence_list

    def load(self):
        print("Loading layer from", self.path, "...")

        real_path = get_pytorch_model(self.path)

        data = torch.load(real_path)

        assert data, f"Could not load registered layer at {real_path}"

        local_sequence_list = data["model_meta_dict"]["topology"]

        assert (
            local_sequence_list
        ), f"Could not load local sequence of the registered layer {real_path}"

        diff = JsonPatch.from_diff(local_sequence_list, self.sequence_list)

        assert (
            len(diff.patch) == 0,
            f"Saved layer {real_path} sequence does not match with provided sequence: "
            "make sure you are providing the good uid and that the sequence is equivalent",
        )

        self.load_state_dict(data["model_state_dict"])
        self.requires_grad_(self.learnable)

    def forward(self, x):
        return self.sequence(x)


def _find_module_of_layertype(module: torch.nn.Module, ltype) -> List[Any]:
    output = []
    for mod in module.children():
        if isinstance(mod, ltype):
            output.append(mod)
        output.extend(_find_module_of_layertype(mod, ltype))
    return output


def load_load_layers(module: torch.nn.Module):
    modlist: List[LoadModule] = _find_module_of_layertype(module, LoadModule)
    for mod in modlist:
        mod.load()


def save_save_layers(module: torch.nn.Module):
    modlist: List[SaveModule] = _find_module_of_layertype(module, SaveModule)
    for mod in modlist:
        mod.save()
