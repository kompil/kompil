import torch
import torch.nn.functional

from torch.nn.modules.module import Module
from torch.nn.functional import relu
from torch._ops import ops

from kompil.nn.layers.quantized.base import _Quantized
from kompil.nn.layers.quantized.utils import sample_tensor, make_cache
from kompil.nn.layers.spish import SPish, spish


class ObservedSPish(Module):
    def __init__(self):
        super().__init__()
        self.weight = None
        self.inputs = None

    def forward(self, x):

        with torch.no_grad():
            y = spish(x, self.weight)

            # Here to avoid out of memory
            sample = sample_tensor(x, 700000).half()

            if self.inputs is None:
                self.inputs = sample
            else:
                self.inputs = torch.cat([sample, torch.flatten(self.inputs)], dim=0)

        return y

    @classmethod
    def from_float(cls, float_spish: SPish):
        obs_spish = cls()
        obs_spish.weight = float_spish.weight

        # qconfig has to be transmitted. If not, pytorch won't know how to quantize the thing.
        if hasattr(float_spish, "qconfig"):
            setattr(obs_spish, "qconfig", getattr(float_spish, "qconfig"))

        return obs_spish

    def extra_repr(self) -> str:
        return f"weight={self.weight[0]}, {self.weight[1]}"


class QuantizedSPish(_Quantized):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.Tensor(2))
        self.register_buffer("cache", torch.Tensor(256))

    def forward(self, x):
        # Keep positive values as it
        pos = relu(x)
        # Negative values
        neg = ops.quantized.mul_scalar(torch.relu(ops.quantized.mul_scalar(x, -1)), -1)
        # Use stored int value as index
        neg_idx = neg.int_repr().long()
        # Get corresponding output value according input index
        pished_neg = self.cache[neg_idx]

        pseudo_spish = ops.quantized.add(pished_neg, pos, self.scale, self.zero_point)

        return pseudo_spish

    @classmethod
    def from_observed(cls, obs_spish: ObservedSPish):
        scale, zero_point = obs_spish.activation_post_process.calculate_qparams()
        inputs = obs_spish.inputs.float().cpu()
        obs_spish.inputs = None  # For garbage collection and avoid OOM

        with torch.no_grad():
            inputs = torch.flatten(torch.nan_to_num(inputs, 0, 0, 0))
            outputs = spish(inputs, obs_spish.weight)

            cache = make_cache(inputs, outputs)

            # Quantize the cache table so the lookup will be straightforward
            qcache = torch.quantize_per_tensor(cache, scale, zero_point, torch.quint8)

        quant_spish = cls()
        quant_spish.weight = obs_spish.weight
        quant_spish.scale = scale
        quant_spish.zero_point = zero_point
        quant_spish.cache = qcache

        return quant_spish

    def extra_repr(self) -> str:
        return f"weight={self.weight}, scale={self.scale}, zero_point={self.zero_point}"

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(QuantizedSPish, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "cache"] = self.cache

    @torch.jit.export
    def __getstate__(self):
        state = super(QuantizedSPish, self).__getstate__()
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

        super(QuantizedSPish, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )

    @torch.jit.export
    def __setstate__(self, state):
        super(QuantizedSPish, self).__setstate__(state)
        self.cache = state[-1]
