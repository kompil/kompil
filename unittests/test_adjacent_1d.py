import unittest
import torch
import math

from kompil.nn.layers.adjacent_1d import Adjacent1d, _CudaFunction
from kompil.utils.numbers import PrimeNumbers

from kompil_ext import (
    cuda_adjacent_1d_forward,
    cuda_adjacent_1d_backward,
)


def build_standard_data(
    i: int, k: int, o: int, ic: int, oc: int, bs: int, get_input: callable, tensor_opts: dict
) -> (torch.Tensor, torch.Tensor):
    """
    Build an example based of layer configuration.

    :param get_input: callable to get the right input id based on the weight id. It is specific to
                      the example needed to be tested, and redondant with adjacent 1d calculation.
    """
    pm = PrimeNumbers()

    # Initialize tensors
    data_in = torch.zeros(bs, ic, i, **tensor_opts)
    weights = torch.zeros(ic, k, oc, o, **tensor_opts)
    bias = torch.zeros(oc, o, **tensor_opts)

    # Feed with prime numbers
    for x_bs in range(bs):
        for x_ic in range(ic):
            for x_i in range(i):
                data_in[x_bs, x_ic, x_i] = pm.pop()

    for x_oc in range(oc):
        for x_o in range(o):
            for x_ic in range(ic):
                for x_k in range(k):
                    weights[x_ic, x_k, x_oc, x_o] = pm.pop()

    for x_oc in range(oc):
        for x_o in range(o):
            bias[x_oc, x_o] = pm.pop()

    # Calculate output
    data_out = torch.zeros(bs, oc, o, **tensor_opts)

    for x_bs in range(bs):
        for x_oc in range(oc):
            for x_o in range(o):
                # Init with Bias
                data_out[x_bs, x_oc, x_o] = bias[x_oc, x_o]
                # Add weight * input using get_input rule
                for x_ic in range(ic):
                    for x_k in range(k):
                        x_i = get_input(x_k, x_o)
                        mult = weights[x_ic, x_k, x_oc, x_o] * data_in[x_bs, x_ic, x_i]
                        data_out[x_bs, x_oc, x_o] += mult

    # Return
    return data_in, weights, bias, data_out


def _get_input_i1k1o2(x_k: int, x_o: int) -> int:
    # output: x_i
    return {
        (0, 0): 0,
        (0, 1): 0,
    }[(x_k, x_o)]


def _get_input_i4k2o2(x_k: int, x_o: int) -> int:
    # output: x_i
    return {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 2,
        (1, 1): 3,
    }[(x_k, x_o)]


def _get_input_i3k2o2(x_k: int, x_o: int) -> int:
    # output: x_i
    return {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 1,
        (1, 1): 2,
    }[(x_k, x_o)]


def _get_input_i4k3o2(x_k: int, x_o: int) -> int:
    # output: x_i
    return {
        (0, 0): 0,
        (1, 0): 1,
        (2, 0): 2,
        (0, 1): 1,
        (1, 1): 2,
        (2, 1): 3,
    }[(x_k, x_o)]


def _get_input_i4k2o6(x_k: int, x_o: int) -> int:
    # output: x_i
    return {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 0,
        (1, 1): 1,
        (0, 2): 1,
        (1, 2): 2,
        (0, 3): 1,
        (1, 3): 2,
        (0, 4): 2,
        (1, 4): 3,
        (0, 5): 2,
        (1, 5): 3,
    }[(x_k, x_o)]


def get_input_generic(i, k, o):
    return {
        (1, 1, 2): _get_input_i1k1o2,
        (4, 2, 2): _get_input_i4k2o2,
        (3, 2, 2): _get_input_i3k2o2,
        (4, 3, 2): _get_input_i4k3o2,
        (4, 2, 6): _get_input_i4k2o6,
    }[(i, k, o)]


class TestForward(unittest.TestCase):
    """
    Test computation for forward operation.
    """

    def setUp(self):
        self.device = torch.device("cuda")
        self.tensor_opts = {"device": self.device, "dtype": torch.float}

    def __std_test_forward(self, data_in, weights, bias, data_out):
        # Get implicit data
        oc = data_out.shape[1]
        o = data_out.shape[2]
        k = weights.shape[1]
        # Forward
        output = cuda_adjacent_1d_forward(data_in, oc, o, k, weights, bias)
        # Test result
        self.assertTrue(torch.all(torch.eq(output, data_out)))

    def __std_gen_test_forward(self, i, k, o, ic, oc, bs):
        getin = get_input_generic(i, k, o)
        self.__std_test_forward(*build_standard_data(i, k, o, ic, oc, bs, getin, self.tensor_opts))

    def test_forward_i1k1o2(self):
        self.__std_gen_test_forward(i=1, k=1, o=2, ic=1, oc=1, bs=1)

    def test_forward_i4k2o2(self):
        self.__std_gen_test_forward(i=4, k=2, o=2, ic=1, oc=1, bs=1)

    def test_forward_i3k2o2(self):
        self.__std_gen_test_forward(i=3, k=2, o=2, ic=1, oc=1, bs=1)

    def test_forward_i4k3o2(self):
        self.__std_gen_test_forward(i=4, k=3, o=2, ic=1, oc=1, bs=1)

    def test_forward_i4k2o6(self):
        self.__std_gen_test_forward(i=4, k=2, o=6, ic=1, oc=1, bs=1)

    def test_after_linear(self):
        batch_size = 32
        # Create layer
        layer0 = torch.nn.Linear(in_features=10, out_features=100)
        layer1 = Adjacent1d(input_dim=(1, 100), output_dim=(1, 100), kernel_size=10, bias=True)

        layer0 = layer0.to(self.device)
        layer1 = layer1.to(self.device)

        # Create data
        data_in = torch.randn(batch_size, 10, **self.tensor_opts)

        # Forward
        x = layer0(data_in)
        x = torch.nn.functional.relu(x)
        x = x.view(batch_size, 1, 100)
        x = layer1(x)


class TestLearn(unittest.TestCase):
    """
    Test effective learning.
    """

    def setUp(self):
        self.device = torch.device("cuda")
        self.tensor_opts = {"device": self.device, "dtype": torch.float}

    def __std_test_learn(self, data_in, weights, bias, data_out):
        # Get implicit data
        oc = data_out.shape[1]
        o = data_out.shape[2]
        k = weights.shape[1]

        # Initialisation
        bias = torch.zeros_like(bias, requires_grad=True)
        weights = torch.randn_like(weights, requires_grad=True)
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(5))

        # Learn
        optimizer = torch.optim.Adam([weights], lr=0.1)
        error = 1
        count = 0
        while error > 1e-4:
            # Timeout on epoch count
            self.assertLess(count, 10000, "Last epoch reached without convergeance")
            count += 1
            # Epoch learning
            prediction = _CudaFunction.apply(data_in, (oc, o), k, weights, bias)
            loss = torch.nn.functional.mse_loss(prediction, data_out)
            error = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def __std_gen_test_learn(self, i, k, o, ic, oc, bs):
        getin = get_input_generic(i, k, o)
        self.__std_test_learn(*build_standard_data(i, k, o, ic, oc, bs, getin, self.tensor_opts))

    def test_learn_i1k1o2(self):
        self.__std_gen_test_learn(i=1, k=1, o=2, ic=1, oc=1, bs=1)

    def test_learn_i4k2o2(self):
        self.__std_gen_test_learn(i=4, k=2, o=2, ic=1, oc=1, bs=1)

    def test_learn_i3k2o2(self):
        self.__std_gen_test_learn(i=3, k=2, o=2, ic=1, oc=1, bs=1)

    def test_learn_i4k3o2(self):
        self.__std_gen_test_learn(i=4, k=3, o=2, ic=1, oc=1, bs=1)

    def test_learn_i4k2o6(self):
        self.__std_gen_test_learn(i=4, k=2, o=6, ic=1, oc=1, bs=1)


class TestLimits(unittest.TestCase):
    """
    Test limits for layer.
    """

    def setUp(self):
        self.device = torch.device("cuda")
        self.tensor_opts = {"device": self.device, "dtype": torch.float}

    def __std_test_iteration(self, bs: int = 2, **kwargs):
        # Create layer
        layer = Adjacent1d(**kwargs).to(self.device)

        # Create data
        input_dim = kwargs["input_dim"]
        output_dim = kwargs["output_dim"]
        data_in = torch.randn(bs, *input_dim, **self.tensor_opts)
        data_out = torch.randn(bs, *output_dim, **self.tensor_opts)

        # Forward and backward
        prediction = layer(data_in)
        loss = torch.nn.functional.mse_loss(prediction, data_out)
        error = loss.item()
        loss.backward()

    def test_data_loss(self):

        layer = Adjacent1d(input_dim=(1, 10), output_dim=(1, 8), kernel_size=2, bias=True)
        layer = Adjacent1d(input_dim=(1, 10), output_dim=(1, 5), kernel_size=2, bias=True)
        with self.assertRaises(Exception):
            layer = Adjacent1d(input_dim=(1, 10), output_dim=(1, 2), kernel_size=2, bias=True)

    def test_dim(self):
        self.__std_test_iteration(input_dim=(1, 100), output_dim=(1, 1000), kernel_size=10)
        self.__std_test_iteration(input_dim=(1, 100), output_dim=(1, 10000000), kernel_size=10)
        self.__std_test_iteration(input_dim=(1, 100), output_dim=(1, 1000000), kernel_size=100)

        for i in range(4):  # Ensure it as it is based on a non deterministic error.
            self.__std_test_iteration(
                input_dim=(1, 150000), output_dim=(1, 150000), kernel_size=1, bs=32
            )


if __name__ == "__main__":
    unittest.main()
