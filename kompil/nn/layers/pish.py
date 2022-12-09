import torch

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import mish


try:
    from kompil_activations import cuda_pish_forward, cuda_pish_backward

    has_kompil_ext = True
except ImportError as exception:
    print("WARNING: kompil pish extension not imported.")
    print("         Using slow and vram intensive implementation instead.")
    has_kompil_ext = False


def pish(x, weight):
    parametrized_neg = weight[0] * x
    bound_neg = torch.maximum(x, parametrized_neg)
    parametrized_mish_and_pos = weight[1] * mish(x)

    y = torch.minimum(parametrized_mish_and_pos, bound_neg)

    return y


class _CudaPishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batch_in: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:

        if not batch_in.is_contiguous():
            batch_in = batch_in.contiguous()

        output = cuda_pish_forward(batch_in, weight)

        ctx.save_for_backward(batch_in, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        batch_in, weight = ctx.saved_tensors

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_input, grad_weight = cuda_pish_backward(batch_in, weight, grad_output)

        return grad_input, grad_weight


class Pish(Module):
    def __init__(self):
        super().__init__()

        self._params = Parameter(torch.Tensor([0.1, 1]))

    def forward(self, x):
        if x.is_cuda and has_kompil_ext:
            return _CudaPishFunction.apply(x, self._params)
        else:
            return pish(x, self._params)
