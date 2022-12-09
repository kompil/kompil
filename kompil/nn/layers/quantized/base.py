import torch
import abc

from torch.nn.modules import Module


class _Quantized(Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self.scale = 1
        self.zero_point = 0

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(_Quantized, self)._save_to_state_dict(destination, prefix, keep_vars)
        if hasattr(self, "weight"):
            destination[prefix + "weight"] = self.weight

        destination[prefix + "scale"] = (
            self.scale if isinstance(self.scale, torch.Tensor) else torch.tensor(self.scale)
        )
        destination[prefix + "zero_point"] = (
            self.zero_point
            if isinstance(self.zero_point, torch.Tensor)
            else torch.tensor(self.zero_point)
        )

    @torch.jit.export
    def __getstate__(self):
        if hasattr(self, "weight"):
            return (
                self.scale,
                self.zero_point,
                self.weight,
            )
        else:
            return (
                self.scale,
                self.zero_point,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if prefix + "weight" in state_dict:
            self.weight = state_dict[prefix + "weight"].data
            state_dict.pop(prefix + "weight")

        if prefix + "scale" in state_dict:
            self.scale = float(state_dict[prefix + "scale"])
            state_dict.pop(prefix + "scale")

        if prefix + "zero_point" in state_dict:
            self.zero_point = int(state_dict[prefix + "zero_point"])
            state_dict.pop(prefix + "zero_point")

        super(_Quantized, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )

    @torch.jit.export
    def __setstate__(self, state):
        self.scale = state[0]
        self.zero_point = state[1]

        if hasattr(self, "weight"):
            self.weight = state[2]
