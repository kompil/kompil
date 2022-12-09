import math
import torch
import enum
import torch.nn.functional

from typing import Tuple, Optional, Union
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function

try:
    from kompil_adjacent import (
        cuda_adjacent_1d_forward,
        cuda_adjacent_1d_backward,
    )

    has_kompil_ext = True
except ImportError as exception:
    print("WARNING: kompil adjacent extension not imported. Some features will be disabled")
    has_kompil_ext = False


class _CudaFunction(Function):
    @staticmethod
    def forward(
        ctx,
        batch_in: Tensor,  # tensor of shape (N, in_c, in_s)
        output_dim: Tuple[int, int],  # out_c, out_s
        kernel_size: int,  # ker
        weights: Tensor,  # tensor of shape (in_c, ker, out_c, out_s)
        bias: Tensor = None,  # tensor of shape (out_c, out_s)
    ) -> Tensor:  # tensor of shape (N, out_c, out_s)

        out_c, out_s = output_dim

        if not batch_in.is_contiguous():
            batch_in = batch_in.contiguous()

        output = cuda_adjacent_1d_forward(batch_in, out_c, out_s, kernel_size, weights, bias)

        ctx.save_for_backward(batch_in, weights, bias)
        ctx.constant = (output_dim, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        batch_in, weights, bias = ctx.saved_tensors
        output_dim, kernel_size = ctx.constant

        out_c, out_s = output_dim

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_input, grad_weights, grad_bias = cuda_adjacent_1d_backward(
            batch_in, out_c, out_s, kernel_size, weights, bias, grad_output
        )

        return grad_input, None, None, grad_weights, grad_bias


class Adjacent1d(Module):
    """
    Torch module for adjacent 1d operation.
    """

    __constants__ = ["input_dim", "output_dim", "kernel_size"]

    def __init__(
        self,
        input_dim: Tuple[int, int],  # in_c, in_s
        output_dim: Tuple[int, int],  # out_c, out_s
        kernel_size: int,  # ker
        bias: bool = True,
    ) -> None:
        assert bias, f"{self.__name__} with no bias is not supported yet"

        out_c, out_s = output_dim
        in_c, in_s = input_dim

        assert in_s / kernel_size <= out_s, "Data loss along the axis"

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        in_c = input_dim[0]

        self.weights = Parameter(torch.Tensor(in_c, kernel_size, out_c, out_s))

        if bias:
            self.bias = Parameter(torch.Tensor(out_c, out_s))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return _CudaFunction.apply(x, self.output_dim, self.kernel_size, self.weights, self.bias)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, kernel_size={self.kernel_size}, bias={self.bias is not None}"
