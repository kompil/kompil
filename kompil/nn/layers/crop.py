import torch

from torch.nn.modules.module import Module


def crop2d_center(tensor: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """Crop an image at the center"""
    # Ensure compat
    assert len(tensor.shape) in [3, 4]
    assert tensor.shape[-1] >= tw
    assert tensor.shape[-2] >= th
    # Calculate first pixel
    h, w = tensor.shape[-2:]
    x1 = int(round((w - tw) / 2.0))
    y1 = int(round((h - th) / 2.0))
    # Return the cropped tensor
    if len(tensor.shape) == 3:
        return tensor[:, y1 : y1 + th, x1 : x1 + tw]
    if len(tensor.shape) == 4:
        return tensor[:, :, y1 : y1 + th, x1 : x1 + tw]


class Crop2d(Module):

    __constants__ = ["__th", "__tw"]

    def __init__(self, th: int, tw: int):
        super().__init__()
        self.__th = th
        self.__tw = tw

    def forward(self, x):
        return crop2d_center(x, self.__th, self.__tw)

    def extra_repr(self) -> str:
        return f"th={self.__th}, tw={self.__tw}"
