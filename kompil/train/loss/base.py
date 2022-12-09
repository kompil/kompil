import types
import torch

from torch.utils.data import IterableDataset

from kompil.utils.factory import Factory

__FACTORY = Factory("loss")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_loss(name: str):
    def to_class(fct):
        if isinstance(fct, types.FunctionType):
            return FunctionLoss(fct)
        return fct()

    return factory().register(name, hook=to_class)


class Loss:
    def __call__(self, y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
        raise NotImplementedError("Loss uncallable")

    def on_bench(self, epoch: int, benchmark, criteria):
        pass

    def prepare(self, dataset: IterableDataset, **kwargs):
        pass


class FunctionLoss(Loss):
    """
    Loss based on a simple function.
    """

    def __init__(self, fct):
        self.__fct = fct
        self.__kwargs = None
        self.__name = f"Loss<{fct.__name__}>"

    def prepare(self, _: IterableDataset, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
        assert self.__kwargs is not None, "Loss is not prepared"
        return self.__fct(y_pred, y_ref, context=context, **self.__kwargs)

    def __str__(self):
        return self.__name
