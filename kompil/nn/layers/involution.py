import torch
from torch import Tensor
from torch.nn.modules.module import Module


class Involution2D(Module):

    __constants__ = [
        "_channels",
        "_groups",
        "_kernel",
        "_stride",
        "_reduction",
        "_dilation",
        "_padding",
    ]

    def __init__(
        self,
        channels: int,
        kernel: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        reduction: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self._channels = channels
        self._groups = groups
        self._kernel = kernel
        self._dilation = dilation
        self._padding = padding
        self._stride = stride
        self._reduction = reduction

        self.o = torch.nn.AvgPool2d(stride, stride) if stride > 1 else torch.nn.Identity()
        self.reduce = torch.nn.Conv2d(channels, channels // reduction, 1)
        self.span = torch.nn.Conv2d(channels // reduction, kernel * kernel * groups, 1)
        self.unfold = torch.nn.Unfold(kernel, dilation, padding, stride)

    def forward(self, x: Tensor) -> Tensor:
        bs, c, h, w = x.shape
        assert c == self._channels
        k2 = self._kernel * self._kernel

        x_unfolded = self.unfold(x)  # B,CxKxK,HxW
        x_unfolded = x_unfolded.view(bs, self._groups, c // self._groups, k2, h, w)
        # kernel generation, Eqn.(6)
        kernel = self.span(self.reduce(self.o(x)))  # B,KxKxG,H,W
        kernel = kernel.view(bs, self._groups, k2, h, w).unsqueeze(2)
        # Multiply-Add operation, Eqn.(4)
        out = torch.mul(kernel, x_unfolded).sum(dim=3)  # B,G,C/G,H,W
        out = out.view(bs, c, h, w)
        return out


class BuilderInvolution2d:
    TYPE = "invol2d"

    @classmethod
    def elem(
        cls,
        kernel: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        reduction: int = 1,
        groups: int = 1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            reduction=reduction,
            groups=groups,
        )

    @classmethod
    def make(
        cls,
        input_size,
        kernel,
        stride,
        dilation,
        padding,
        reduction,
        groups,
        **kwargs,
    ) -> Module:
        return Involution2D(
            input_size[0],
            kernel,
            stride,
            dilation,
            padding,
            reduction,
            groups,
        )

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        assert isinstance(input_size, tuple)
        assert len(input_size) == 3
        return input_size
