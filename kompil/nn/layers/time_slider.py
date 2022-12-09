import torch
import torch.nn.functional

from torch import Tensor
from torch.nn.modules.module import Module
from torch.autograd import Function


class _Function(Function):
    @staticmethod
    def forward(
        ctx,
        batch_in: Tensor,  # tensor of shape (batch_size, 1)
        window: int,
    ) -> Tensor:  # tensor of shape (N, time_frame, 1)

        ctx.constant = batch_in.shape

        batch_size, _ = batch_in.shape

        output = torch.empty(batch_size, window, 1, device=batch_in.device, dtype=batch_in.dtype)

        for batch in range(batch_size):
            base_idframe = batch_in[batch].long().item()

            for t in range(window):
                idframe = base_idframe + t
                output[batch][t][0] = idframe

        return output

    @staticmethod
    def backward(ctx, grad_output):

        batch_in_shape = ctx.constant

        # Nothing has to be learned
        grad_input = torch.zeros(batch_in_shape, device=grad_output.device, dtype=grad_output.dtype)

        return grad_input, None


class TimeSlider(Module):
    """
    Module that provide a list of gray codes [n, n+1, ... n+k] based on the gray code "n"
    """

    __constants__ = ["window"]

    def __init__(self, window: int):
        super().__init__()
        self.window = window

    def forward(self, x: Tensor) -> Tensor:
        return _Function.apply(x, self.window)

    def extra_repr(self) -> str:
        return f"window={self.window}"
