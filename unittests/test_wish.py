import unittest
import torch
import math

from kompil.nn.layers.wish import _python_wish_std, _CudaWishFunction
from kompil.utils.numbers import PrimeNumbers

from kompil_activations import cuda_wish_forward, cuda_wish_backward


class Test(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.datatype = torch.half

    def test_forward(self):

        batch_in = torch.linspace(-3.0, 3.0, 1000, device=self.device)

        for i in range(100):
            weight = torch.randn(2, device=self.device, dtype=self.datatype)

            output_custom = cuda_wish_forward(batch_in, weight)

            output_python = _python_wish_std(batch_in, weight)

            for i in range(batch_in.numel()):
                self.assertAlmostEqual(
                    output_custom[i].item(),
                    output_python[i].item(),
                    msg=f"i: {batch_in[i].item()}, w0: {weight[0].item()}, w1: {weight[1].item()}",
                    delta=1e-5,
                )


if __name__ == "__main__":
    unittest.main()
