import unittest
import torch
import math

from kompil.nn.layers.pish import _python_cuda_forward, _CudaPishFunction
from kompil.utils.numbers import PrimeNumbers

from kompil_pish import cuda_pish_forward, cuda_pish_backward


class Test(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")

    def test_forward(self):

        for i in range(100):
            batch_in = torch.randn(4, 6, 160, 240, device=self.device)
            weight = torch.randn(2, device=self.device)

            output_custom = cuda_pish_forward(batch_in, weight)

            output_python = _python_cuda_forward(batch_in, weight)

            self.assertTrue(torch.all(torch.eq(output_custom, output_python)))

    def test_backward(self):
        class ModulePython(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight.clone())

            def forward(self, x):
                return _python_cuda_forward(x, self.weight)

        class ModuleCustom(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight.clone())

            def forward(self, x):
                return _CudaPishFunction.apply(x, self.weight)

        for i in range(100):
            batch_in = torch.randn(4, 6, 160, 240, device=self.device)
            target_output = torch.randn_like(batch_in)
            weight = torch.randn(2, device=self.device)

            module_python = ModulePython(weight)
            module_custom = ModuleCustom(weight)

            python_batch_in = torch.nn.Parameter(batch_in.clone())
            python_output = module_python(python_batch_in)
            loss = (python_output - target_output).sum()
            loss.backward()

            custom_batch_in = torch.nn.Parameter(batch_in.clone())
            custom_output = module_custom(custom_batch_in)
            loss = (custom_output - target_output).sum()
            loss.backward()

            w1_eq = torch.eq(module_python.weight.grad.data[0], module_custom.weight.grad.data[0])

            self.assertTrue(torch.all(w1_eq), msg="Weight 1 are not the same")

            w2_eq = torch.eq(module_python.weight.grad.data[1], module_custom.weight.grad.data[1])

            self.assertTrue(torch.all(w2_eq), msg="Weight 2 are not the same")

            self.assertTrue(
                torch.all(torch.eq(python_batch_in.grad.data, custom_batch_in.grad.data)),
                msg="grad input are not the same",
            )


if __name__ == "__main__":
    unittest.main()
