import torch

from torch.nn.modules.module import Module
from torch.nn.utils import weight_norm


class SubBlockConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        bias: bool,
        apply_weight_norm: bool,
    ):
        """
        Build up an image from smaller images.
        Small images are stored into channels, then these images are concatenate to make the full image.
        Works like subpixel, but with block.
        Example : input 192,40,60 ==x8==> output 3,320,480

        """
        super().__init__()

        assert scale_factor > 0

        self.scale_factor = scale_factor

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * self.scale_factor**2,
            kernel_size=1,  # No kernel; compute must be done in previous layers
            bias=bias,
        )

        if apply_weight_norm:
            self.conv = weight_norm(self.conv)

    def forward(self, x):
        y = self.conv(x)

        b, c_i, h_i, w_i = y.shape
        factor_sqr = self.scale_factor**2

        c_o = int(c_i / factor_sqr)
        h_o = int(h_i * self.scale_factor)
        w_o = int(w_i * self.scale_factor)

        full = torch.empty(size=(b, c_o, h_o, w_o), dtype=y.dtype, device=y.device)

        for c_i_idx in range(c_i):
            c_i_idx_rel = c_i_idx % factor_sqr

            c_o_idx = int(c_i_idx / factor_sqr)
            h_o_idx = int((c_i_idx_rel * h_i) / h_o) * h_i
            w_o_idx = (c_i_idx_rel * w_i) % w_o

            h_o_end = h_o_idx + h_i
            w_o_end = w_o_idx + w_i

            # Put the entire block to this location
            full[:, c_o_idx, h_o_idx:h_o_end, w_o_idx:w_o_end] = y[:, c_i_idx]

        return full
