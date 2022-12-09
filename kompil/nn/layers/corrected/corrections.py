import torch
import abc

from typing import Union
from torch.nn import Module
from torch._ops import ops


class _Correction(Module, abc.ABC):
    def __init__(self):
        super(_Correction, self).__init__()

        self.register_buffer("data", torch.Tensor([]))

    @classmethod
    def from_data(cls, quant_module: Module, output_errors: torch.Tensor) -> Module:
        raise NotImplementedError()

    @abc.abstractmethod
    def pack(self, data: torch.Tensor, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def unpack(self, idx: Union[int, slice, None], **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def apply_correction(self, x: torch.Tensor, idx: Union[int, slice]) -> torch.Tensor:
        raise NotImplementedError()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(_Correction, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "data"] = self.data

    @torch.jit.export
    def __getstate__(self):
        return (self.data,)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        self.data = state_dict[prefix + "data"]
        state_dict.pop(prefix + "data")

        super(_Correction, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )

    @torch.jit.export
    def __setstate__(self, state):
        self.data = state[0]


class DummyCorrection(_Correction):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_data(cls, quant_module: Module, output_errors: torch.Tensor) -> Module:
        correction_obj = cls()
        correction_obj.pack(
            output_errors, scale=quant_module.scale, zero_point=quant_module.zero_point
        )

        return correction_obj

    def pack(self, errors: torch.Tensor, **kwargs):
        scale = kwargs["scale"]
        zp = kwargs["zero_point"]

        # Quantize the error
        qerrors = torch.quantize_per_tensor(errors, scale, zp, dtype=torch.quint8)

        # Store as remnant data
        self.data = qerrors

    def unpack(self, idx, **kwargs) -> torch.Tensor:
        return self.data[idx]

    def apply_correction(self, x: torch.Tensor, idx: Union[int, slice]) -> torch.Tensor:
        qcorr = self.unpack(idx)

        corrected_output = ops.quantized.add(x, qcorr, x.q_scale(), x.q_zero_point())

        return corrected_output
