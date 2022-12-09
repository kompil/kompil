import torch
from typing import List
from torch.nn.modules.module import Module


class BuilderPixelShuffle:
    """Module kept for retrocompatibility, use the more generic ChannelUpscale instead"""

    TYPE = "pixel_shuffle"

    @classmethod
    def elem(cls, factor: int) -> dict:
        return dict(type=cls.TYPE, factor=factor)

    @classmethod
    def make(cls, factor, **kwargs) -> Module:
        return torch.nn.PixelShuffle(factor)

    @classmethod
    def predict_size(cls, input_size, factor, **kwargs) -> tuple:
        c_in, h_in, w_in = input_size[-3:]

        ouput_size = tuple(
            [
                *input_size[:-3],
                int(c_in / factor / factor),
                h_in * factor,
                w_in * factor,
            ]
        )

        return ouput_size


def channel_upscale(data: torch.Tensor, upscale_factors: List[int]) -> torch.Tensor:
    """
    channel_upscale operation is the same as the pytorch pixel shuffle but generic for different
    dimensions (2D, 3D and so on...) with different upscale factors.

    Right now, only the standard 2D with the same factor is supported.
    """
    # general compat
    assert (
        len(data.shape) == len(upscale_factors) + 2
    ), "Channel upscale need factors dim == 2 + input dim"
    # Compute output channels
    chan_dim = -len(upscale_factors) - 1
    channels = data.shape[chan_dim]
    out_chan = int(channels)
    for factor in upscale_factors:
        out_chan = out_chan // factor
    # Compute target dims
    batch_size = data.shape[0]
    dims_to_upscale = data.shape[-len(upscale_factors) :]
    target_dims = [
        batch_size,
        out_chan,
        *[dims_to_upscale[i] * upscale_factors[i] for i in range(len(upscale_factors))],
    ]
    # Compute permute order
    dim_start_1 = 2
    dim_start_2 = dim_start_1 + len(upscale_factors)
    permute_order = [0, 1]
    for i in range(len(upscale_factors)):
        permute_order.append(dim_start_2 + i)
        permute_order.append(dim_start_1 + i)
    # Explode, permute and merge
    exploded_dims = [batch_size, out_chan, *upscale_factors, *dims_to_upscale]
    reviewed = data.contiguous().view(*exploded_dims)
    permuted = reviewed.permute(*permute_order).contiguous()
    return permuted.view(*target_dims)


class ChannelUpscale(Module):
    __constants__ = ["__upscale_factors"]

    def __init__(self, upscale_factors: List[int]) -> None:
        super().__init__()
        self.__upscale_factors = upscale_factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return channel_upscale(x, self.__upscale_factors)

    def extra_repr(self) -> str:
        return ", ".join(str(f) for f in self.__upscale_factors)


class BuilderChannelUpscale:
    TYPE = "channel_upscale"

    @classmethod
    def elem(cls, upscale_factors: List[int]) -> dict:
        return dict(type=cls.TYPE, upscale_factors=upscale_factors)

    @classmethod
    def make(cls, upscale_factors, **kwargs) -> Module:
        return ChannelUpscale(upscale_factors)

    @classmethod
    def predict_size(cls, input_size, upscale_factors, **kwargs) -> tuple:
        chan_dim = -len(upscale_factors) - 1

        kept_dims = input_size[:chan_dim]
        channels = input_size[chan_dim]
        upscaled_dims = list(input_size[chan_dim + 1 :])

        for dim, factor in enumerate(upscale_factors):
            revert_dim = len(input_size) - dim
            if channels % factor != 0:
                raise RuntimeError(
                    f"Channels ({channels} / {input_size[chan_dim]}) not divisible by {factor} "
                    f"on dimension {revert_dim}"
                )
            channels = channels // factor
            upscaled_dims[dim] = upscaled_dims[dim] * factor

        return tuple([*kept_dims, channels, *upscaled_dims])
