#!/usr/bin/env python3
import sys
import time
import torch
import typing
from kompil.nn.layers import Adjacent2d, Method

DEVICE = torch.device("cuda")
BATCH_SIZE = 32


class Adj2DOpt:
    def __init__(self, **kwargs):
        self.input: typing.Tuple[int, int, int] = kwargs.get("input")
        self.output: typing.Tuple[int, int, int] = kwargs.get("output")
        self.kernel: typing.Tuple[int, int] = kwargs.get("kernel")
        self.bias: bool = kwargs.get("bias")

    def generate(self, acceleration: Method = Method.CUDA) -> Adjacent2d:
        return Adjacent2d(self.input, self.output, self.kernel, self.bias, acceleration)

    def numel(self):
        outputs = self.output[0] * self.output[1] * self.output[2]
        inputs = self.kernel[0] * self.kernel[1] * self.input[0]
        if self.bias:
            return outputs * inputs + outputs
        return outputs * inputs


class LinearOpt:
    def __init__(self, **kwargs):
        self.in_features: int = kwargs.get("in_features")
        self.out_features: int = kwargs.get("out_features")
        self.bias: bool = kwargs.get("bias")

    def generate(self) -> torch.nn.Linear:
        return torch.nn.Linear(self.in_features, self.out_features, bias=self.bias)

    def numel(self):
        matcount = self.in_features * self.out_features
        if self.bias:
            return matcount + self.out_features
        return matcount


def equivalent_adj2d_to_linear(adj2d_opt: Adj2DOpt) -> LinearOpt:
    out_features = adj2d_opt.output[0] * adj2d_opt.output[1] * adj2d_opt.output[2]
    in_features = adj2d_opt.kernel[0] * adj2d_opt.kernel[1] * adj2d_opt.input[0]
    bias = True
    return LinearOpt(in_features=in_features, out_features=out_features, bias=bias)


def bench_linear(opt: LinearOpt, batch_size: int, ite: int = 1):
    print("#" * 10)
    print("#", bench_linear.__name__)
    print()
    # Input
    batch_in = torch.randn(batch_size, opt.in_features, device=DEVICE)
    # Layer
    layer = opt.generate()
    layer = layer.to(DEVICE)
    # Run
    torch.cuda.synchronize()
    wanted_res = torch.zeros(batch_size, opt.out_features, device=DEVICE)
    forward_total_time = 0
    backward_total_time = 0
    for _ in range(ite):
        # Forward
        start = time.time()
        res = layer(batch_in)
        torch.cuda.synchronize()
        forward_total_time += time.time() - start
        # Loss
        loss = torch.nn.functional.mse_loss(res, wanted_res)
        # Backward
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_total_time += time.time() - start
    # Print
    mean_forward = forward_total_time / ite
    mean_backward = backward_total_time / ite
    mean_ite_per_sec = 1.0 / (mean_forward + mean_backward)
    print("in_features:", opt.in_features)
    print("out_features:", opt.out_features)
    print("bias:", opt.bias)
    print("device:", DEVICE)
    print("iterations:", ite)
    print("batch size:", batch_size)
    print()
    print("parameters:", opt.numel())
    print("mean forward time:", mean_forward, "seconds")
    print("mean backward time:", mean_backward, "seconds")
    print("learning speed impact:", mean_ite_per_sec, "it/s")
    print()


def bench_adjacent_2d(
    opt: Adj2DOpt, batch_size: int, acc: Method, ite: int = 1, backward: bool = False
):
    print("#" * 10)
    print("#", bench_adjacent_2d.__name__, acc)
    print()
    # Input
    batch_in = torch.randn(batch_size, *opt.input, device=DEVICE)
    # Layer
    creation_start = time.time()
    layer = opt.generate(acc).to(DEVICE)
    torch.cuda.synchronize()
    creation_time = time.time() - creation_start
    # Run
    torch.cuda.synchronize()
    wanted_res = torch.zeros(batch_size, *opt.output, device=DEVICE)
    forward_total_time = 0
    backward_total_time = 0
    for _ in range(ite):
        # Forward
        start = time.time()
        res = layer(batch_in)
        torch.cuda.synchronize()
        forward_total_time += time.time() - start
        if not backward:
            continue
        # Loss
        loss = torch.nn.functional.mse_loss(res, wanted_res)
        # Backward
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_total_time += time.time() - start
    # Print
    mean_forward = forward_total_time / ite
    mean_backward = backward_total_time / ite
    mean_ite_per_sec = 1.0 / (mean_forward + mean_backward)
    print("acceleration:", acc)
    print("batch size:", batch_size)
    print("input:", opt.input)
    print("output:", opt.output)
    print("kernel:", opt.kernel)
    print("device:", DEVICE)
    print("bias:", True)
    print("iterations:", ite)
    print()
    print("parameters:", opt.numel())
    print("creation time:", creation_time, "seconds")
    print("mean forward time:", mean_forward, "seconds")
    print("mean backward time:", mean_backward, "seconds")
    print("it predicted speed:", mean_ite_per_sec, "it/s")
    print()


def test():
    opts = [
        Adj2DOpt(
            input=(3, 320, 480),
            output=(3, 320, 480),
            kernel=(6, 6),
            bias=True,
        ),
        Adj2DOpt(
            input=(1, 320, 480),
            output=(1, 320, 480),
            kernel=(6, 6),
            bias=True,
        ),
        Adj2DOpt(
            input=(3, 320, 480),
            output=(3, 320, 480),
            kernel=(1, 1),
            bias=True,
        ),
    ]

    for opt in opts:
        bench_adjacent_2d(opt, batch_size=BATCH_SIZE, acc=Method.CUDA, ite=10, backward=True)
        if "linear" in sys.argv[1:]:
            bench_linear(equivalent_adj2d_to_linear(opt), batch_size=BATCH_SIZE, ite=10)
        if "cpp" in sys.argv[1:]:
            bench_adjacent_2d(opt, batch_size=BATCH_SIZE, acc=Method.CPP, ite=1, backward=True)
        if "python" in sys.argv[1:]:
            bench_adjacent_2d(opt, batch_size=BATCH_SIZE, acc=Method.PYTHON, ite=1, backward=True)


if __name__ == "__main__":
    torch.cuda.init()
    test()
