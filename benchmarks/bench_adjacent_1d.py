#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import sys
import time
import torch
import typing
import argparse
import argcomplete
import tabulate
from kompil.nn.layers.adjacent_1d import Adjacent1d
from kompil.utils.time import MeanTimer
from kompil.utils.numbers import to_scale

DEVICE = torch.device("cuda")


def title(msg: str):
    print("#" * 10)
    print("#", msg)


class BenchResult:
    def __init__(self):
        self.creation_timer = MeanTimer()
        self.forward_timer = MeanTimer()
        self.backward_timer = MeanTimer()
        self.n_parameters = None


class RefResult:
    def __init__(self):
        self.forward_timer = MeanTimer()
        self.backward_timer = MeanTimer()


class Bench:
    def __init__(self, bs: int, ic: int, i: int, oc: int, o: int, ker: int):
        self.values = (bs, ic, i, oc, o, ker)
        self.name = f"i{ic}x{i}:k{ker}:o{oc}x{o}"

    def calculate_ref(self, iterations: int) -> RefResult:
        bs, ic, _, oc, o, ker = self.values
        # Hard data
        ite = iterations
        device = DEVICE
        # Result
        result = RefResult()
        # Layer
        in_features = ker * ic
        out_features = o * oc
        layer = torch.nn.Linear(in_features, out_features, bias=True)
        layer = layer.to(device)
        torch.cuda.synchronize()
        # Create data
        batch_in = torch.randn(bs, in_features, device=device)
        batch_out = torch.randn(bs, out_features, device=device)
        # Run
        for _ in range(ite):
            # Forward
            with result.forward_timer:
                res = layer(batch_in)
                torch.cuda.synchronize()
            # Loss
            loss = torch.nn.functional.mse_loss(res, batch_out)
            # Backward
            with result.backward_timer:
                loss.backward()
                torch.cuda.synchronize()
        # Return result
        return result

    def bench(self, iterations: int):
        bs, ic, i, oc, o, ker = self.values
        ite = iterations
        # Hard data
        device = DEVICE
        bias = True

        # Create data
        batch_in = torch.randn(bs, ic, i, device=device)
        batch_out = torch.randn(bs, oc, o, device=device)

        # Result
        result = BenchResult()

        # Create layer
        with result.creation_timer:
            layer = Adjacent1d(input_dim=(ic, i), output_dim=(oc, o), kernel_size=ker, bias=bias)
            layer = layer.to(device)
            torch.cuda.synchronize()

        # Print parameters
        result.n_parameters = layer.weights.numel() + layer.bias.numel()

        # Run
        for _ in range(ite):
            # Forward
            with result.forward_timer:
                res = layer(batch_in)
                torch.cuda.synchronize()
            # Loss
            loss = torch.nn.functional.mse_loss(res, batch_out)
            # Backward
            with result.backward_timer:
                loss.backward()
                torch.cuda.synchronize()

        # Return
        return result

    def __internal_bench(self, table: bool, iterations: int):
        bs, ic, i, oc, o, ker = self.values
        if not table:
            title(f"Bench {self.name} (BS {bs})")
        # Reference
        ref_res = self.calculate_ref(iterations)
        # Bench
        bench_res = self.bench(iterations)
        # Additional calculations
        total_mean_time = bench_res.forward_timer.mean_time + bench_res.backward_timer.mean_time
        mean_ite_per_sec = 1.0 / total_mean_time
        relative_forward_time = ref_res.forward_timer.mean_time / bench_res.forward_timer.mean_time
        relative_backward_time = (
            ref_res.backward_timer.mean_time / bench_res.backward_timer.mean_time
        )
        # Return
        if table:
            return [
                f"{to_scale(bench_res.n_parameters)}Param",
                f"{to_scale(ref_res.forward_timer.mean_time)}s",
                f"{to_scale(ref_res.backward_timer.mean_time)}s",
                f"{to_scale(bench_res.forward_timer.mean_time)}s",
                f"{to_scale(bench_res.backward_timer.mean_time)}s",
                f"{100.0 * relative_forward_time:0.2f}%",
                f"{100.0 * relative_backward_time:0.2f}%",
            ]

        print(
            "reference:",
            f"f: {to_scale(ref_res.forward_timer.mean_time)}s; "
            f"b: {to_scale(ref_res.backward_timer.mean_time)}s",
        )
        print(f"it predicted speed: {mean_ite_per_sec:0.2f} it/s")
        print(
            "results:",
            f"f: {to_scale(bench_res.forward_timer.mean_time)}s; "
            f"b: {to_scale(bench_res.backward_timer.mean_time)}s",
        )
        print(
            "efficiency:",
            f"f: {100.0 * relative_forward_time:0.2f}%; "
            f"b: {100.0 * relative_backward_time:0.2f}%",
        )
        print()

    def bench_to_table(self, iterations: int):
        return self.__internal_bench(True, iterations)

    def print_bench(self, iterations: int):
        return self.__internal_bench(False, iterations)


def to_web_browser(html_content: str):
    import webbrowser
    import tempfile

    fp = tempfile.NamedTemporaryFile(prefix="bench_", suffix=".html", mode="w+", delete=False)
    path = fp.name
    fp.write(html_content)
    fp.close()

    webbrowser.open(path, new=2)


def test(print_table: str, iterations: int):
    benches = [
        # Test variation batch size
        Bench(bs=32, ic=10, i=10000, oc=10, o=10000, ker=10),
        Bench(bs=16, ic=10, i=10000, oc=10, o=10000, ker=10),
        Bench(bs=8, ic=10, i=10000, oc=10, o=10000, ker=10),
        Bench(bs=4, ic=10, i=10000, oc=10, o=10000, ker=10),
        Bench(bs=1, ic=10, i=10000, oc=10, o=10000, ker=10),
        # Test variation input and output
        Bench(bs=16, ic=3, i=10000, oc=30, o=10000, ker=10),
        Bench(bs=16, ic=3, i=1000, oc=30, o=10000, ker=10),
        Bench(bs=16, ic=3, i=10000, oc=30, o=1000, ker=10),
        # Test variation output channel
        Bench(bs=1, ic=1, i=100000, oc=1, o=100000, ker=1),
    ]
    if print_table == "full":
        for bench in benches:
            bench.print_bench(iterations)
        return

    table = [[*bench.values, *bench.bench_to_table(iterations)] for bench in benches]
    headers = [
        "bs",
        "ic",
        "i",
        "oc",
        "o",
        "ker",
        "Params",
        "Ref fwd",
        "Ref bwd",
        "Bnch fwd",
        "Bnch bwd",
        "Eff fwd",
        "Eff bwd",
    ]

    if print_table == "table":
        print(tabulate.tabulate(table, headers, tablefmt="fancy_grid"))
    else:
        to_web_browser(tabulate.tabulate(table, headers, tablefmt="html"))


def main():
    parser = argparse.ArgumentParser(description="Bench for adjacent1d layer")
    parser.set_defaults(func=lambda _: parser.print_help())
    parser.add_argument("--type", type=str, choices=["table", "full", "html"], default="table")
    parser.add_argument("-i", "--iterations", type=int, default=3)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    torch.cuda.init()

    test(args.type, args.iterations)


if __name__ == "__main__":
    main()
