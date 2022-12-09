import abc
import torch

from kompil.utils.factory import Factory


class Metric(abc.ABC):
    @abc.abstractproperty
    def frame_count(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def add_frame(self, ref: torch.Tensor, dist: torch.Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def score_at_index(self, index: int) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_score_list(self) -> torch.Tensor:
        raise NotImplementedError()


class MetricEmptyShell(Metric):
    HIGHER_IS_BETTER = True

    def __init__(self):
        self.__counter = 0

    @property
    def frame_count(self) -> int:
        return self.__counter

    def add_frame(self, ref: torch.Tensor, dist: torch.Tensor):
        self.__counter += 1

    def compute(self) -> float:
        pass

    def score_at_index(self, index: int) -> float:
        return 0.0

    def get_score_list(self) -> torch.Tensor:
        return torch.zeros(self.__counter, dtype=torch.float64)


__FACTORY = Factory("metrics")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_metric(name: str, *args, **kwargs):
    def decorator(cls):
        nonlocal name, args, kwargs
        assert issubclass(cls, Metric), f"{cls} is not a metric"
        factory().register_item(name=name, cls=(cls, args, kwargs))
        return cls

    return decorator


def metric_higher_is_better(name: str) -> bool:
    return factory().get(name)[0].HIGHER_IS_BETTER
