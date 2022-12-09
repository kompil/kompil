import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from kompil.nn.layers import Adjacent2d, Method


class RandTest:
    def __init__(self, batch_in_shape, out_shape, kernel_shape):
        self.batch_in_shape = batch_in_shape
        self.out_shape = out_shape
        self.kernel_shape = kernel_shape

        batch_size, in_c, in_h, in_w = batch_in_shape
        out_c, out_h, out_w = out_shape
        ker_h, ker_w = kernel_shape

        self.batch_in = torch.randn(*batch_in_shape, requires_grad=True)
        self.weights = torch.randn(in_c, ker_h, ker_w, out_c, out_h, out_w)
        self.bias = torch.randn(*out_shape)


def get_max_diff(tensor1, tensor2):
    return torch.abs(tensor1 - tensor2).max()


class TestAdjacent2d(unittest.TestCase):
    def setUp(self):
        self.__test_0 = RandTest(
            batch_in_shape=(9, 1, 4, 4),
            out_shape=(3, 6, 6),
            kernel_shape=(2, 2),
        )

    def __run_test(self, test, device, acceleration: Method):
        batch_size = test.batch_in_shape[0]

        model = Adjacent2d(
            test.batch_in_shape[1:],
            test.out_shape,
            test.kernel_shape,
            bias=True,
            acceleration=acceleration,
        ).to(device)

        model.weights.data = test.weights.to(device)
        model.bias.data = test.bias.to(device)

        prediction = model(test.batch_in.to(device))

        self.assertEqual(prediction.shape, torch.Size([batch_size, *test.out_shape]))

        expected = torch.zeros_like(prediction)

        loss = torch.nn.functional.mse_loss(prediction, expected)

        loss.backward()

        gradients = (
            test.batch_in.grad.clone(),
            model.weights.grad.clone(),
            model.bias.grad.clone(),
        )
        test.batch_in.grad.fill_(0.0)
        model.weights.grad.fill_(0.0)
        model.bias.grad.fill_(0.0)

        return prediction, gradients

    def test_run_on_cpu(self):
        device = torch.device("cpu")
        output_python, gradients_python = self.__run_test(self.__test_0, device, Method.PYTHON)
        output_cpp, gradients_cpp = self.__run_test(self.__test_0, device, Method.CPP)

        self.assertTrue(torch.allclose(output_python, output_cpp))

        for elem in range(len(gradients_python)):
            self.assertTrue(torch.allclose(gradients_python[elem], gradients_cpp[elem]))

    def test_run_on_cuda(self):
        device = torch.device("cuda")
        output_python, gradients_python = self.__run_test(self.__test_0, device, Method.PYTHON)
        output_cpp, gradients_cpp = self.__run_test(self.__test_0, device, Method.CPP)
        output_cuda, gradients_cuda = self.__run_test(self.__test_0, device, Method.CUDA)

        self.assertTrue(torch.allclose(output_python, output_cpp))

        max_diff = get_max_diff(output_cpp, output_cuda)
        if max_diff > 1e-5:
            raise Exception(f"Cuda and CPP results does not match: max diff {max_diff}")

        for elem in range(len(gradients_python)):
            self.assertTrue(torch.allclose(gradients_python[elem], gradients_cpp[elem]))

            max_diff = get_max_diff(gradients_cpp[elem], gradients_cuda[elem])
            if max_diff > 1e-5:
                raise Exception(f"Cuda and CPP results does not match: max diff {max_diff}")

    def test_values(self):
        batch_size, in_c, in_h, in_w = 1, 1, 3, 3
        out_c, out_h, out_w = 3, 2, 2
        output_dim = (out_c, out_h, out_w)
        ker_h, ker_w = 2, 2
        kernel_size = (ker_h, ker_w)

        batch_in = torch.FloatTensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])

        model = Adjacent2d((in_c, in_h, in_w), output_dim, kernel_size)

        model.weights.data = torch.zeros(*model.weights.data.shape)

        model.weights.data[0, 0, 1, 0, 0, 0] = 1.0
        model.weights.data[0, 0, 0, 1, 0, 0] = 1.0

        model.weights.data[0, 0, 1, 0, 0, 1] = 2.0
        model.weights.data[0, 0, 0, 1, 0, 1] = 2.0

        model.bias.data = torch.zeros(out_c, out_h, out_w)

        output = model(batch_in)

        expected_tensor = torch.Tensor(
            [[[2.0, 6.0], [0.0, 0.0]], [[1.0, 4.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
        )

        self.assertTrue(torch.all(torch.eq(output[0], expected_tensor)))

    def __test_learning(self, device, acceleration):
        batch_size, in_c, in_h, in_w = 10, 1, 3, 3
        input_dim = (in_c, in_h, in_w)
        out_c, out_h, out_w = 1, 3, 3
        output_dim = (out_c, out_h, out_w)
        ker_h, ker_w = 2, 2
        kernel_size = (ker_h, ker_w)

        batch_in = torch.randn(batch_size, in_c, in_h, in_w)
        batch_out = batch_in * 2.0

        dataset = TensorDataset(batch_in, batch_out)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        class VideoNet(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = Adjacent2d(
                    input_dim, output_dim, kernel_size, bias=True, acceleration=acceleration
                )
                self.layer.train()

            def forward(self, x):
                return self.__common_forward(x)

            def training_step(self, batch, batch_idx):
                x, y_ref = batch
                y_pred = self.layer(x)
                return torch.nn.functional.mse_loss(y_pred, y_ref)

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.1)

        model = VideoNet().to(device)

        trainer = pl.Trainer(
            max_epochs=10,
            gpus=1 if device.type == "cuda" else 0,
        )

        x, y = dataset[0]
        batch_0 = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        loss_before = model.training_step(batch_0, 0).item()
        weights_before = model.layer.weights.data.clone()
        bias_before = model.layer.bias.data.clone()

        trainer.fit(model, dataloader)

        model = model.to(device)

        loss_after = model.training_step(batch_0, 0).item()
        weights_after = model.layer.weights.data.clone()
        bias_after = model.layer.bias.data.clone()

        self.assertFalse(torch.any(torch.eq(weights_before, weights_after)))
        self.assertFalse(torch.any(torch.eq(bias_before, bias_after)))
        self.assertTrue(loss_after < loss_before)

    def test_learning(self):
        """
        Try learn with every methods.
        """
        self.__test_learning(torch.device("cpu"), Method.PYTHON)
        self.__test_learning(torch.device("cpu"), Method.CPP)
        self.__test_learning(torch.device("cuda"), Method.PYTHON)
        self.__test_learning(torch.device("cuda"), Method.CPP)
        self.__test_learning(torch.device("cuda"), Method.CUDA)

    def test_cuda_chain_learning(self):

        # Data
        device = torch.device("cuda")
        batch_size = 8
        input_dim = (100,)
        intermediate_dim = (1, 96, 160)
        output_dim = (3, 480, 720)
        kernel = (4, 6)
        batch_in = torch.randn(batch_size, *input_dim).to(device)
        target_output = torch.randn(batch_size, *output_dim).to(device)
        # Layer
        class TestLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                intermediate_nodes = int(
                    intermediate_dim[0] * intermediate_dim[1] * intermediate_dim[2]
                )
                self.linear = torch.nn.Linear(input_dim[0], intermediate_nodes)
                self.prelu = torch.nn.PReLU()
                self.adj2d_0 = Adjacent2d(
                    intermediate_dim, intermediate_dim, kernel, True, Method.CUDA
                )
                self.prelu2 = torch.nn.PReLU()
                self.adj2d_1 = Adjacent2d(intermediate_dim, output_dim, kernel, True, Method.CUDA)

            def forward(self, x):
                x = self.linear(x)
                x = self.prelu(x)
                x = x.view(batch_size, *intermediate_dim)
                x = self.adj2d_0(x)
                x = self.prelu2(x)
                x = self.adj2d_1(x)
                return x

        layer = TestLayer().to(device)
        # Run
        for _ in range(2):
            res = layer(batch_in)
            loss = torch.nn.functional.mse_loss(res, target_output)
            loss.backward()


if __name__ == "__main__":
    unittest.main()
