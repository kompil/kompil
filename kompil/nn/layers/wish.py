import torch

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import mish


try:
    from kompil_activations import cuda_wish_forward, cuda_wish_backward

    has_kompil_ext = True
except ImportError as exception:
    print("WARNING: kompil wish extension not imported.")
    print("         Using slow and vram intensive implementation instead.")
    has_kompil_ext = False


def _python_wish(x, weight, gate):
    return torch.where(x < gate, (x - gate) * weight[0], torch.zeros_like(x)) + weight[1] * mish(x)


class _CudaWishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batch_in: torch.Tensor, weight: torch.Tensor, gate: float) -> torch.Tensor:

        if not batch_in.is_contiguous():
            batch_in = batch_in.contiguous()

        output = cuda_wish_forward(batch_in, weight, gate)

        ctx.save_for_backward(batch_in, weight)
        ctx.constant = gate

        return output

    @staticmethod
    def backward(ctx, grad_output):
        batch_in, weight = ctx.saved_tensors
        gate = ctx.constant

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_input, grad_weight = cuda_wish_backward(batch_in, weight, grad_output, gate)

        return grad_input, grad_weight, None


if has_kompil_ext:
    wish_forward = _CudaWishFunction.apply
else:
    wish_forward = _python_wish


class Wish(Module):
    """
    Wish is an activation function based on the following equation:
    if x < gate: (x - gate) * w0 + mish(x) * w1
    else: mish(x) * w1
    """

    def __init__(self, gate: float = -1.0):
        super().__init__()
        self.gate = gate
        self.weight = Parameter(torch.Tensor([0.1, 1]))

    def forward(self, x):
        return wish_forward(x, self.weight, self.gate)

    def extra_repr(self) -> str:
        return str(self.gate)
