import torch

from typing import Dict
from torch.nn import Module

from kompil.nn.layers.quantized.base import _Quantized
from kompil.nn.layers.corrected.corrections import _Correction


class _Corrected(_Quantized):
    correction: _Correction
    context: Dict

    @classmethod
    def from_quantized(cls, quant_mod: Module, correction: _Correction) -> Module:
        raise NotImplementedError()

    def _apply_correction(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.context["frames_idx"].squeeze(0)
        return self.correction.apply_correction(x, idx)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(_Corrected, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "correction_type"] = type(self.correction)

    @torch.jit.export
    def __getstate__(self):
        state = super(_Corrected, self).__getstate__()
        return (*state, self.correction)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        corr_type = state_dict[prefix + "correction_type"]
        state_dict.pop(prefix + "correction_type")
        self.correction = corr_type()

        super(_Corrected, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )

    @torch.jit.export
    def __setstate__(self, state):
        super(_Corrected, self).__setstate__(state)
        self.correction = state[-1]
