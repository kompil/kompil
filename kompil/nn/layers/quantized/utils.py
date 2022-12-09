import torch
import torch.nn.functional

from torch.ao.quantization.observer import MinMaxObserver


def sample_tensor(tensor: torch.Tensor, max_sample: int):
    flat = torch.flatten(tensor)

    if tensor.numel() < max_sample:
        idx = torch.arange(0, tensor.numel())
    else:
        idx = torch.rand(max_sample) * tensor.numel()

    idx = idx.long().to(tensor.device)
    crop = flat.index_select(dim=0, index=idx)

    return crop


def make_cache(inputs: torch.Tensor, outputs: torch.Tensor, size: int = 256) -> torch.Tensor:
    r"""
    Makes a cache table where the output values are indexed regarding the int representation of the quantized inputs
    """
    with torch.no_grad():
        # Make a quant version of the inputs to get all further indexes (which are based on int_repr of the qinput)
        obs = MinMaxObserver()
        obs(inputs)
        input_scale, input_zp = obs.calculate_qparams()
        qinputs = torch.quantize_per_tensor(inputs, input_scale, input_zp, torch.quint8)
        qinput_intr = qinputs.int_repr().long()

        # Make cache table regarding this logic : func(x) = cache(qx.int_repr)
        cache = torch.Tensor(size).fill_(0)
        cache = torch.scatter(input=cache, dim=0, index=qinput_intr, src=outputs)

    return cache
