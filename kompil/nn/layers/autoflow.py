import math
import torch
import torch.nn.functional

from typing import Union
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Function

import kompil.data.timeline as tl
from kompil.utils.numbers import to_scale


class _Function(Function):
    @staticmethod
    def forward(
        ctx,
        batch_in: Tensor,  # tensor of shape (N, *, 1)
        data: Tensor,  # tensor of shape (nb_frames, *output_shape)
    ) -> Tensor:  # tensor of shape (N, *, *output_shape)

        batch_size = batch_in.shape[0]
        inter_size = torch.Size(batch_in.shape[1:-1])
        output_shape = torch.Size(data.shape[1:])

        # Flat the batch_size and optional dimensions, treat them likewise
        batch_in_flatten = batch_in.view(-1, 1)

        # For each elements, select the right data
        flatten_size = batch_in_flatten.shape[0]
        output = torch.empty(
            flatten_size, *output_shape, device=batch_in.device, dtype=batch_in.dtype
        )

        for idx in range(flatten_size):
            idframe = batch_in_flatten[idx].long().item()
            output[idx].copy_(data[idframe])

        # Restore the batches and optional dimensions
        output = output.view(batch_size, *inter_size, *output_shape)

        ctx.save_for_backward(batch_in, data)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        batch_in, data = ctx.saved_tensors

        output_shape = torch.Size(data.shape[1:])

        # Flat the batch_size and optional dimensions, treat them likewise
        batch_in_flatten = batch_in.view(-1, 1)
        grad_output_flatten = grad_output.view(-1, *output_shape)

        # Add the grad output to the right data
        grad_data = torch.zeros_like(data)

        flatten_size = batch_in_flatten.shape[0]
        for idx in range(flatten_size):
            idframe = batch_in_flatten[idx].long().item()
            grad_data[idframe].add_(grad_output_flatten[idx])

        # Input is not propagated
        grad_input = torch.zeros_like(batch_in)

        return grad_input, grad_data


def _to_shape(shape: Union[int, torch.Size, tuple]) -> torch.Size:
    actual_shape = shape
    if isinstance(shape, tuple):
        actual_shape = torch.Size(shape)
    if isinstance(shape, int):
        actual_shape = torch.Size((shape,))
    return actual_shape


class AutoFlow(Module):
    """
    Torch module for direct application of latent state
    """

    __constants__ = ["output_shape", "nb_data"]

    def __init__(self, nb_data: int, output_shape: Union[int, torch.Size, tuple]):
        super().__init__()
        self.output_shape = _to_shape(output_shape)
        self.nb_data = nb_data

        self.data = Parameter(torch.Tensor(nb_data, *self.output_shape))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.data, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return _Function.apply(x, self.data)

    def extra_repr(self) -> str:
        return (
            f"nb_data={self.nb_data}, output_shape={self.output_shape}, "
            f"params={to_scale(self.data.data.numel())}"
        )
