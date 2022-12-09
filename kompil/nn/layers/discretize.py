import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.autograd import Function
from kompil.utils.video import discretize
from kompil.utils.colorspace import yuv420_to_yuv420n, yuv420n_to_yuv420


class _FunctionDiscretize(Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        return discretize(input)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


class Discretize(Module):
    __constants__ = ["__colorspace"]

    __CLRSPCES = ["rgb8", "yuv420"]

    def __init__(self, colorspace: str):
        assert colorspace in self.__CLRSPCES, f"discretize colorspace unknown: {self.__colorspace}"
        super().__init__()
        self.__colorspace = colorspace

    def forward(self, x: Tensor) -> Tensor:
        if self.__colorspace == "yuv420":
            x = yuv420_to_yuv420n(x)
        x = _FunctionDiscretize.apply(x)
        if self.__colorspace == "yuv420":
            x = yuv420n_to_yuv420(x)
        return x


class BuilderDiscretize:
    TYPE = "discretize"

    @classmethod
    def elem(cls, colorspace: str = "rgb8") -> dict:
        return dict(type=cls.TYPE, colorspace=colorspace)

    @classmethod
    def make(cls, colorspace, **kwargs) -> Module:
        return Discretize(colorspace=colorspace)

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


class DiscretizedPReLU(Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        # Apply PReLU over all data
        x_prelu = torch.clamp(self.prelu(x), max=0.0)

        # Apply discretization on [0; 1] values
        x_disc = _FunctionDiscretize.apply(torch.clamp(x, min=0.0, max=1.0))

        # Apply same PReLU learnable param over [1; +inf[ values
        x_up_rect = -torch.clamp(-(x - 1), max=0.0) * self.prelu.weight[0]

        return x_prelu + x_disc + x_up_rect


class BuilderDiscretizedPReLU:
    TYPE = "drelu"

    @classmethod
    def elem(cls) -> dict:
        return dict(type=cls.TYPE)

    @classmethod
    def make(cls, **kwargs) -> Module:
        return DiscretizedPReLU()

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size


def plot_test():
    import plotly.graph_objects as go

    x = torch.linspace(-2.0, 2.0, dtype=torch.float, steps=1000)

    d = DiscretizedPReLU()

    y = d(x)

    xn = x.detach().numpy()
    yn = y.detach().numpy()

    scatters = [go.Scatter(name="psnr_min", x=xn, y=yn, line=go.Line(color="green"))]

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="x"),
            yaxis=dict(title="y"),
        )
    )

    fig = go.Figure(data=scatters, layout=layout)
    fig.show()


if __name__ == "__main__":
    plot_test()
