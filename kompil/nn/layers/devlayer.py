import torch
from typing import Dict, Any, Tuple


ERROR_MSG = """Devlayer is a temporary empty shell for the main branch.
If you want to infere on a trained model, find the branch where the model has been trained.
"""


class DevLayer(torch.nn.Module):
    """
    Devlayer is an empty shell that can be modified to quickly test some unusual topologies.
    The goal is to never merge any topology in the main branch so it still send an error
    message we try to run a model trained with it.
    """

    def __init__(self, context: Dict[str, Any], nb_frames: int, params: list, outshape: Tuple[int]):
        super().__init__()

    def forward(self, _):
        raise NotImplementedError(ERROR_MSG)


class BuilderDevLayer:
    TYPE = "devlayer"

    @classmethod
    def elem(cls, nb_frames: int, params: list, outshape: Tuple[int]) -> dict:
        return dict(type=cls.TYPE, nb_frames=nb_frames, params=params, outshape=outshape)

    @classmethod
    def make(cls, context, nb_frames, params, outshape, **kwargs) -> torch.nn.Module:
        return DevLayer(context, nb_frames, params, outshape)

    @classmethod
    def predict_size(cls, outshape, **kwargs) -> tuple:
        return outshape
