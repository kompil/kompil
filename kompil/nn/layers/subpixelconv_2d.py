import torch

from torch.nn.modules.module import Module


class SubPixelConv2d(Module):
    def __init__(
        self,
        kernel_size: tuple,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        stride: tuple,
        dilation: tuple,
        padding: tuple,
        padding_mode: str,
        bias: bool,
    ):
        """
        Based on https://arxiv.org/pdf/1609.05158.pdf
        """

        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * scale_factor**2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))
