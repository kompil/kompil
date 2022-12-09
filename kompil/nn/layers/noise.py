import torch

from torch.nn.modules.module import Module


class NormalNoise(Module):
    def __init__(self, mean: float, std: float):
        super().__init__()

        self._std = std
        self._mean = mean

    def forward(self, x):
        if self.training:
            if self._mean == 0:
                mean = torch.zeros_like(x)
            else:
                mean = torch.ones_like(x) * self._mean

            normal = torch.normal(mean, torch.ones_like(x) * self._std)

            return x + normal
        else:
            return x

    def extra_repr(self) -> str:
        return f"mean={self._mean}, std={self._std}"


class BuilderNoise:
    TYPE = "noise"

    @classmethod
    def normal(cls, mean: float = 0.0, std: float = 0.01) -> dict:
        return dict(type=cls.TYPE, distribution="normal", mean=mean, std=std)

    @classmethod
    def make(cls, input_size, distribution: str, mean: float, std: float, **kwargs) -> Module:
        if distribution == "normal":
            return NormalNoise(mean, std)
        else:
            raise NotImplementedError(f"Unknown distribution {distribution}")

    @classmethod
    def predict_size(cls, input_size, **kwargs) -> tuple:
        return input_size
