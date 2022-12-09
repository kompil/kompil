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
        adjacent_2d_forward,
        adjacent_2d_backward,
        cuda_adjacent_2d_forward,
        cuda_adjacent_2d_backward,
    )

    has_kompil_ext = True
except ImportError as exception:
    print("WARNING: kompil adjacent extension not imported. Some features will be disabled")
    has_kompil_ext = False


def _each_pixels(width, height, stride_x=1, stride_y=1) -> Tuple[int, int]:
    for x in range(0, width, stride_x):
        for y in range(0, height, stride_y):
            yield x, y


class _Adjacent2dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        batch_in: Tensor,  # tensor of shape (N, in_c, in_h, in_w)
        output_dim: Tuple[int, int, int],  # out_c, out_h, out_w
        kernel_size: Tuple[int, int],  # ker_h, ker_w
        weights: Tensor,  # tensor of shape (in_c, ker_h, ker_w, out_c, out_h, out_w)
        bias: Tensor = None,  # tensor of shape (out_c, out_h, out_w)
    ) -> Tensor:  # tensor of shape (N, out_c, out_h, out_w)

        # Shortcuts
        batch_size, in_c, in_h, in_w = batch_in.shape

        out_c, out_h, out_w = output_dim

        ker_h, ker_w = kernel_size
        batch_size, in_c, in_h, in_w = batch_in.shape

        device = batch_in.device

        # Create an input data to be used in the final operation
        flat_ker_in_dim = in_c * ker_h * ker_w
        tmp_in = torch.empty(out_h, out_w, batch_size, in_c, ker_h, ker_w, device=device)

        # Calculate stride
        in_w_red = in_w - ker_w + 1
        in_h_red = in_h - ker_h + 1

        in_step_x = float(in_w - ker_w) / (float(out_w) - 1)
        in_step_y = float(in_h - ker_h) / (float(out_h) - 1)

        # Iterate over output pixels
        for x, y in _each_pixels(out_w, out_h):
            in_start_x = round(float(x) * in_step_x)
            in_end_x = in_start_x + ker_w
            in_start_y = round(float(y) * in_step_y)
            in_end_y = in_start_y + ker_h

            local_in = batch_in[:, :, in_start_y:in_end_y, in_start_x:in_end_x]

            tmp_in[y, x] = local_in

        # Calculate the input * weight
        tmp_in = tmp_in.view(out_h * out_w, batch_size, flat_ker_in_dim)
        weights_reorganized = weights.view(flat_ker_in_dim, out_c, out_h * out_w)
        weights_reorganized = weights_reorganized.permute(2, 0, 1)

        res = torch.empty(out_h * out_w, batch_size, out_c, device=device)
        torch.bmm(tmp_in, weights_reorganized, out=res)

        # Init output with bias
        output = torch.empty(batch_size, out_c, out_h, out_w, device=device)
        for i in range(batch_size):
            output[i] = bias

        # Add the result to the output
        res = res.permute(1, 2, 0)
        output.add_(res.view(batch_size, out_c, out_h, out_w))

        ctx.save_for_backward(batch_in, weights, bias)
        ctx.constant = (output_dim, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        batch_in, weights, bias = ctx.saved_tensors
        output_dim, kernel_size = ctx.constant

        # Init gradients
        grad_input = torch.zeros(*batch_in.shape, device=batch_in.device)
        grad_weights = torch.zeros(*weights.shape, device=batch_in.device)
        grad_bias = torch.zeros(*bias.shape, device=batch_in.device)

        # Get working dimensions
        batch_size, in_c, in_h, in_w = batch_in.shape

        out_c, out_h, out_w = output_dim

        ker_h, ker_w = kernel_size

        in_step_x = float(in_w - ker_w) / (float(out_w) - 1)
        in_step_y = float(in_h - ker_h) / (float(out_h) - 1)

        # For retrocomp, flat the weights
        flat_ker_in_dim = in_c * ker_h * ker_w
        grad_weights = grad_weights.view(flat_ker_in_dim, out_c, out_h, out_w)
        weights = weights.view(flat_ker_in_dim, out_c, out_h, out_w)

        # Iterate over output pixels to apply the kernel.
        for x, y in _each_pixels(out_w, out_h):
            # Extract input kernel and weights
            in_start_x = round(float(x) * in_step_x)
            in_end_x = in_start_x + ker_w
            in_start_y = round(float(y) * in_step_y)
            in_end_y = in_start_y + ker_h

            local_in = batch_in[:, :, in_start_y:in_end_y, in_start_x:in_end_x]
            local_weights = weights[:, :, y, x].t()

            # Right now there is an issue with view() for splitted tensors, this
            # fix the issue but also instanciate undesired memory.
            local_in = local_in.clone()

            # Set weights such that it matches the local_in tensor
            flat_dimension = ker_w * ker_h * in_c
            local_in = local_in.view(batch_size, flat_dimension)

            # Calculate the gradients
            local_grad_output = grad_output[:, :, y, x]

            # Inputs gradients
            local_grad_input = local_grad_output.mm(local_weights)
            local_grad_input = local_grad_input.view(batch_size, in_c, ker_h, ker_w)
            grad_input[:, :, in_start_y:in_end_y, in_start_x:in_end_x] += local_grad_input
            # Weights gradients
            local_grad_weights = local_grad_output.t().mm(local_in)
            grad_weights[:, :, y, x] = local_grad_weights.t()
            # Bias gradients
            grad_bias[:, y, x] = local_grad_output.sum(0)

        # For retrocomp, unflat the grad weight
        grad_weights = grad_weights.view(in_c, ker_h, ker_w, out_c, out_h, out_w)

        return grad_input, None, None, grad_weights, grad_bias


class _CppAdjacent2dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        batch_in: Tensor,  # tensor of shape (N, in_c, in_h, in_w)
        output_dim: Tuple[int, int, int],  # out_c, out_h, out_w
        kernel_size: Tuple[int, int],  # ker_h, ker_w
        weights: Tensor,  # tensor of shape (out_c, out_h, out_w, in_c * ker_h * ker_w)
        bias: Tensor = None,  # tensor of shape (out_c, out_h, out_w)
    ) -> Tensor:  # tensor of shape (N, out_c, out_h, out_w)

        out_c, out_h, out_w = output_dim

        ker_h, ker_w = kernel_size

        output = adjacent_2d_forward(batch_in, out_c, out_h, out_w, ker_h, ker_w, weights, bias)

        ctx.save_for_backward(batch_in, weights, bias)
        ctx.constant = (output_dim, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        batch_in, weights, bias = ctx.saved_tensors
        output_dim, kernel_size = ctx.constant

        out_c, out_h, out_w = output_dim
        ker_h, ker_w = kernel_size

        grad_input, grad_weights, grad_bias = adjacent_2d_backward(
            batch_in, out_c, out_h, out_w, ker_h, ker_w, weights, bias, grad_output
        )

        return grad_input, None, None, grad_weights, grad_bias


class _CudaAdjacent2dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        batch_in: Tensor,  # tensor of shape (N, in_c, in_h, in_w)
        output_dim: Tuple[int, int, int],  # out_c, out_h, out_w
        kernel_size: Tuple[int, int],  # ker_h, ker_w
        weights: Tensor,  # tensor of shape (in_c, ker_h, ker_w, out_c, out_h, out_w)
        bias: Tensor = None,  # tensor of shape (out_c, out_h, out_w)
    ) -> Tensor:  # tensor of shape (N, out_c, out_h, out_w)

        out_c, out_h, out_w = output_dim

        ker_h, ker_w = kernel_size

        if not batch_in.is_contiguous():
            batch_in = batch_in.contiguous()

        output = cuda_adjacent_2d_forward(
            batch_in, out_c, out_h, out_w, ker_h, ker_w, weights, bias
        )

        ctx.save_for_backward(batch_in, weights, bias)
        ctx.constant = (output_dim, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        batch_in, weights, bias = ctx.saved_tensors
        output_dim, kernel_size = ctx.constant

        out_c, out_h, out_w = output_dim
        ker_h, ker_w = kernel_size

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_input, grad_weights, grad_bias = cuda_adjacent_2d_backward(
            batch_in, out_c, out_h, out_w, ker_h, ker_w, weights, bias, grad_output
        )

        return grad_input, None, None, grad_weights, grad_bias


class Method(enum.Enum):
    PYTHON = "python"
    CPP = "cpp"
    CUDA = "cuda"


class Adjacent2d(Module):
    """
    Torch module for adjacent 2d operation.
    """

    __constants__ = ["input_dim", "output_dim", "kernel_size"]

    def __init__(
        self,
        input_dim: Tuple[int, int, int],  # in_c, in_h, in_w
        output_dim: Tuple[int, int, int],  # out_c, out_h, out_w
        kernel_size: Tuple[int, int],  # k_h, k_w
        bias: bool = True,
        acceleration: Method = Method.CPP,
    ) -> None:
        assert bias, f"{self.__name__} with no bias is not supported yet"

        out_c, out_h, out_w = output_dim
        ker_h, ker_w = kernel_size
        in_c, in_h, in_w = input_dim

        assert in_w / ker_w <= out_w, "Data loss along the x axis"
        assert in_h / ker_h <= out_h, "Data loss along the y axis"

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        in_c = input_dim[0]

        self.weights = Parameter(torch.Tensor(in_c, ker_h, ker_w, out_c, out_h, out_w))

        if bias:
            self.bias = Parameter(torch.Tensor(out_c, out_h, out_w))
        else:
            self.register_parameter("bias", None)

        self.__function = {
            Method.CPP: _CppAdjacent2dFunction,
            Method.PYTHON: _Adjacent2dFunction,
            Method.CUDA: _CudaAdjacent2dFunction,
        }[acceleration]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return self.__function.apply(x, self.output_dim, self.kernel_size, self.weights, self.bias)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, kernel_size={self.kernel_size}, bias={self.bias is not None}"
