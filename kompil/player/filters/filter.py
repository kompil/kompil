import abc
import torch
from argparse import ArgumentError
from typing import List

from kompil.utils.factory import Factory
from kompil.player.options import PlayerTargetOptions

__FACTORY = Factory("player-filters")


def filter_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_filter(name: str):
    return filter_factory().register(name)


class Filter(abc.ABC):
    @abc.abstractmethod
    def filter(self, image: torch.Tensor, colorspace: str) -> torch.Tensor:
        pass

    def set_slider(self, ratio: float):
        pass

    @property
    def is_slider(self) -> bool:
        return False


def create_filter(data: List[str], opt: PlayerTargetOptions):
    # Default
    if len(data) == 0:
        data = ["rgb"]
    # Get filter
    filter_cls = filter_factory().get(data[0])
    if filter_cls is None:
        raise ArgumentError("unknown filder")
    # Instance it
    return filter_cls.generate(data[1:], opt)
