import math
import torch
from torch import Tensor
from torch.nn.modules.module import Module

from kompil.utils.cpp_utils import (
    graycode,
    get_gc_nodes,
    feed_binary_tensor,
    layer_graycode,
    layer_binary,
)


class EyeInput(Module):

    __constants__ = ["__frames"]

    def __init__(self, nb_frames: int):
        super().__init__()
        self.__frames = nb_frames
        self.__eye = torch.eye(nb_frames)

    def forward(self, x: Tensor) -> Tensor:
        self.__eye = self.__eye.to(x.device)
        self.__eye = self.__eye.to(x.dtype)
        bs = x.shape[0]
        inter_size = torch.Size(x.shape[1:-1])
        x_flat = x.view(-1, 1)
        out_val = torch.cat([self.__eye[x_flat[i].long()] for i in range(x_flat.shape[0])], dim=0)
        return out_val.view(bs, *inter_size, self.__frames)


class BuilderEyeInput:
    TYPE = "in_eye"

    @classmethod
    def elem(cls, nb_frames: int) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames)

    @classmethod
    def make(cls, nb_frames: int, **kwargs) -> Module:
        return EyeInput(nb_frames)

    @classmethod
    def predict_size(cls, input_size, nb_frames, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return nb_frames
        return (*input_size[:-1], nb_frames)


class GrayCodeInput(Module):

    __constants__ = ["__nodes"]

    def __init__(self, nb_frames: int):
        super().__init__()
        self.__nodes = get_gc_nodes(nb_frames)

    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.view(-1, 1)
        y = torch.empty(x_flat.shape[0], self.__nodes, dtype=x.dtype, device=x.device)
        layer_graycode(x_flat, y)
        return y.view(*x.shape[:-1], self.__nodes)

    def extra_repr(self) -> str:
        return f"nodes={self.__nodes}"


class BuilderGrayCodeInput:
    TYPE = "in_gc"

    @classmethod
    def elem(cls, nb_frames: int) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames)

    @classmethod
    def make(cls, nb_frames: int, **kwargs) -> Module:
        return GrayCodeInput(nb_frames)

    @classmethod
    def predict_size(cls, input_size, nb_frames, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return get_gc_nodes(nb_frames)
        return (*input_size[:-1], get_gc_nodes(nb_frames))


class IncrementalGrayCodeInput(Module):

    __constants__ = ["__nodes", "__nb_frames", "__alpha"]

    def __init__(self, nb_frames: int, alpha: float):
        super().__init__()
        self.__nodes = get_gc_nodes(nb_frames)
        self.__nb_frames = nb_frames
        self.__alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.view(-1, 1)
        y = torch.empty(x_flat.shape[0], self.__nodes, dtype=x.dtype, device=x.device)
        layer_graycode(x_flat, y)
        y = y * (1 + x_flat.repeat(1, self.__nodes) * self.__alpha)
        return y.view(*x.shape[:-1], self.__nodes)

    def extra_repr(self) -> str:
        return f"nodes={self.__nodes}, alpha={self.__alpha}"


class BuilderIncrementalGrayCodeInput:
    TYPE = "in_igc"

    @classmethod
    def elem(cls, nb_frames: int, alpha: float = 1.1) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames, alpha=alpha)

    @classmethod
    def make(cls, nb_frames: int, alpha: float, **kwargs) -> Module:
        return IncrementalGrayCodeInput(nb_frames, alpha)

    @classmethod
    def predict_size(cls, input_size, nb_frames, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return get_gc_nodes(nb_frames)
        return (*input_size[:-1], get_gc_nodes(nb_frames))


class BinaryInput(Module):

    __constants__ = ["__nodes"]

    def __init__(self, nb_frames: int):
        super().__init__()
        self.__nodes = get_gc_nodes(nb_frames)

    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.view(-1, 1)
        y_flat = torch.empty(x_flat.shape[0], self.__nodes, dtype=x.dtype, device=x.device)
        layer_binary(x_flat, y_flat)
        return y_flat.view(*x.shape[:-1], self.__nodes)

    def extra_repr(self) -> str:
        return f"nodes={self.__nodes}"


class BuilderBinaryInput:
    TYPE = "in_bin"

    @classmethod
    def elem(cls, nb_frames: int) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames)

    @classmethod
    def make(cls, nb_frames: int, **kwargs) -> Module:
        return BinaryInput(nb_frames)

    @classmethod
    def predict_size(cls, input_size, nb_frames, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return get_gc_nodes(nb_frames)
        return (*input_size[:-1], get_gc_nodes(nb_frames))


class HybridInput(Module):

    __constants__ = ["__nodes", "__nb_frames"]

    def __init__(self, nb_frames: int):
        super().__init__()
        self.__nb_frames = nb_frames
        self.__nodes = get_gc_nodes(nb_frames) + 1

    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.view(-1, 1)
        y_flat = torch.empty(x_flat.shape[0], self.__nodes - 1, dtype=x.dtype, device=x.device)
        layer_graycode(x_flat, y_flat)
        y_flat = torch.cat([x_flat / self.__nb_frames * 4, y_flat], dim=1)
        return y_flat.view(*x.shape[:-1], self.__nodes)

    def extra_repr(self) -> str:
        return f"nodes={self.__nodes}"


class BuilderHybridInput:
    TYPE = "in_hybrid"

    @classmethod
    def elem(cls, nb_frames: int) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames)

    @classmethod
    def make(cls, nb_frames: int, **kwargs) -> Module:
        return HybridInput(nb_frames)

    @classmethod
    def predict_size(cls, input_size, nb_frames, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return get_gc_nodes(nb_frames) + 1
        return (*input_size[:-1], get_gc_nodes(nb_frames) + 1)


class BinarySinusInput(Module):

    __constants__ = ["__nodes", "__nb_frames"]

    def __init__(self, nb_frames: int):
        super().__init__()
        self.__nb_frames = nb_frames
        self.__nodes = get_gc_nodes(nb_frames)

    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.view(-1, 1)
        y_flat = torch.cat([x_flat / math.pow(2, i) for i in range(self.__nodes)], dim=1)
        y_flat = torch.sin(y_flat * 3.14 - 3.14 / 2) / 2 + 0.5
        return y_flat.view(*x.shape[:-1], self.__nodes)

    def extra_repr(self) -> str:
        return f"nodes={self.__nodes}"


class BuilderBinarySinusInput:
    TYPE = "in_binsin"

    @classmethod
    def elem(cls, nb_frames: int) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames)

    @classmethod
    def make(cls, nb_frames: int, **kwargs) -> Module:
        return BinarySinusInput(nb_frames)

    @classmethod
    def predict_size(cls, input_size, nb_frames, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return get_gc_nodes(nb_frames)
        return (*input_size[:-1], get_gc_nodes(nb_frames))


class BinaryTriangleInput(Module):

    __constants__ = ["__nodes", "__nb_frames"]

    def __init__(self, nb_frames: int):
        super().__init__()
        self.__nb_frames = nb_frames
        self.__nodes = get_gc_nodes(nb_frames)

    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.view(-1, 1)
        y_flat = torch.cat([x_flat / math.pow(2, i) for i in range(self.__nodes)], dim=1)
        y_flat = torch.abs((y_flat / 2 - torch.floor(y_flat / 2)) * 2 - 1)
        return y_flat.view(*x.shape[:-1], self.__nodes)

    def extra_repr(self) -> str:
        return f"nodes={self.__nodes}"


class BuilderBinaryTriangleInput:
    TYPE = "in_tmp"

    @classmethod
    def elem(cls, nb_frames: int) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames)

    @classmethod
    def make(cls, nb_frames: int, **kwargs) -> Module:
        return BinaryTriangleInput(nb_frames)

    @classmethod
    def predict_size(cls, input_size, nb_frames, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return get_gc_nodes(nb_frames)
        return (*input_size[:-1], get_gc_nodes(nb_frames))


class BinaryToothInput(Module):

    __constants__ = ["__nodes", "__nb_frames"]

    def __init__(self, nb_frames: int):
        super().__init__()
        self.__nb_frames = nb_frames
        self.__nodes = get_gc_nodes(nb_frames)

    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.view(-1, 1)
        y_flat = torch.cat([x_flat / math.pow(2, i) for i in range(self.__nodes)], dim=1)
        y_flat = (y_flat / 2 - torch.floor(y_flat / 2)) * 2
        return y_flat.view(*x.shape[:-1], self.__nodes)

    def extra_repr(self) -> str:
        return f"nodes={self.__nodes}"


class BuilderBinaryToothInput:
    TYPE = "in_bintooth"

    @classmethod
    def elem(cls, nb_frames: int) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames)

    @classmethod
    def make(cls, nb_frames: int, **kwargs) -> Module:
        return BinaryToothInput(nb_frames)

    @classmethod
    def predict_size(cls, input_size, nb_frames, **kwargs) -> tuple:
        if isinstance(input_size, int):
            return get_gc_nodes(nb_frames)
        return (*input_size[:-1], get_gc_nodes(nb_frames))


class RegularizedGrayCodeInput(Module):
    """
    Input that combine the following technics over a standard gray code input:
    - Start at frame 1
    - Normalisation of the input
    - Scaling up the input
    - Rounding the input
    """

    __constants__ = ["__nodes"]

    def __init__(self, capacity: int):
        super().__init__()
        self.__nodes = get_gc_nodes(capacity + 1)

    def forward(self, x: Tensor) -> Tensor:
        # Flat the input to handle multiple dimensions.
        x_flat = x.view(-1, 1)
        # Start at to avoid huge pressure on the frame 0.
        x_flat = x_flat + 1
        # Create the base graycode data.
        y = torch.empty(x_flat.shape[0], self.__nodes, dtype=x.dtype, device=x.device)
        layer_graycode(x_flat, y)
        # Normalisation of the input to remove pressure on the first layer.
        # There is still some high variations sometime that causes post-quantisation quality
        # drops at certains frames.
        sum = torch.sum(y, dim=1)
        sum = torch.where(sum == 0, torch.ones_like(sum), sum)
        y = y / sum.view(-1, 1)
        # Scaling up the input
        y *= 255
        # Rounding input to be ready for quantisation.
        y.round_()
        # Reshape to the target shape
        return y.view(*x.shape[:-1], self.__nodes)

    def extra_repr(self) -> str:
        return f"nodes={self.__nodes}"


class BuilderRegularizedGrayCodeInput:
    TYPE = "in_regularized_gc"

    @classmethod
    def elem(cls, capacity: int) -> dict:
        return dict(type=cls.TYPE, capacity=capacity)

    @classmethod
    def make(cls, input_size, capacity: int, **kwargs) -> Module:
        assert input_size[-1] == 1
        return RegularizedGrayCodeInput(capacity)

    @classmethod
    def predict_size(cls, input_size, capacity, **kwargs) -> tuple:
        assert input_size[-1] == 1
        if isinstance(input_size, int):
            return get_gc_nodes(capacity)
        return (*input_size[:-1], get_gc_nodes(capacity))
