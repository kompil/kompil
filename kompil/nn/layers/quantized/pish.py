import torch
import torch.nn.functional

from torch.nn.modules.module import Module
from torch._ops import ops

from kompil.nn.layers.quantized.base import _Quantized
from kompil.nn.layers.quantized.utils import make_cache, sample_tensor
from kompil.nn.layers.pish import Pish, pish


class ObservedPish(Module):
    def __init__(self):
        super().__init__()
        self.weight = None
        self.inputs = None

    def forward(self, x):

        with torch.no_grad():
            y = pish(x, self.weight)

            # Here to avoid out of memory
            sample = sample_tensor(x, 700000).half()

            if self.inputs is None:
                self.inputs = sample
            else:
                self.inputs = torch.cat([sample, torch.flatten(self.inputs)], dim=0)

        return y

    @classmethod
    def from_float(cls, float_pish: Pish):
        obs_pish = cls()
        obs_pish.weight = float_pish._params

        # qconfig has to be transmitted. If not, pytorch won't know how to quantize the thing.
        if hasattr(float_pish, "qconfig"):
            setattr(obs_pish, "qconfig", getattr(float_pish, "qconfig"))

        return obs_pish

    def extra_repr(self) -> str:
        return f"weight={self.weight[0]}, {self.weight[1]}"


class QuantizedPish(_Quantized):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.Tensor(2))
        self.register_buffer("cache", torch.Tensor(256))

    def forward(self, x):
        # Use stored int value as index
        idx = x.int_repr().long()

        # Get corresponding output value according input index
        pseudo_pish = self.cache[idx]

        return pseudo_pish

    @classmethod
    def from_observed(cls, obs_pish: ObservedPish):
        scale, zero_point = obs_pish.activation_post_process.calculate_qparams()
        inputs = obs_pish.inputs.float().cpu()
        obs_pish.inputs = None  # For garbage collection and avoid OOM

        with torch.no_grad():
            # Clear inputs and gen outputs
            inputs = torch.flatten(torch.nan_to_num(inputs, 0, 0, 0))
            outputs = pish(inputs, obs_pish.weight)

            # Make cache table regarding this logic : pish(x) = cache(qx.int_repr)
            cache = make_cache(inputs, outputs, size=256)

            # Quantize the cache table so the lookup->result will be straightforward
            qcache = torch.quantize_per_tensor(cache, scale, zero_point, torch.quint8)

        quant_pish = cls()
        quant_pish.weight = obs_pish.weight
        quant_pish.scale = scale
        quant_pish.zero_point = zero_point
        quant_pish.cache = qcache

        return quant_pish

    def extra_repr(self) -> str:
        return f"weight={self.weight}, scale={self.scale}, zero_point={self.zero_point}"

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(QuantizedPish, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "cache"] = self.cache

    @torch.jit.export
    def __getstate__(self):
        state = super(QuantizedPish, self).__getstate__()
        return (
            *state,
            self.cache,
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if prefix + "cache" in state_dict:
            self.cache = state_dict[prefix + "cache"]
            state_dict.pop(prefix + "cache")

        super(QuantizedPish, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )

    @torch.jit.export
    def __setstate__(self, state):
        super(QuantizedPish, self).__setstate__(state)
        self.cache = state[-1]
