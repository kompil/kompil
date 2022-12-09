import math
import torch
import os

from kompil.utils.factory import Factory

__FACTORY = Factory("learning-rate")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_lr(name: str):
    return factory().register(name)


@register_lr("megatron")
def megatron_learning_rate(nb_frames: int, frame_shape: tuple, batch_size: int) -> float:
    """
    nb  lr (bs = 32)
    600 0.00837
    1800 0.00432
    3000 0.00317
    4200 0.00259
    5400 0.00222
    6600 0.00197
    7800 0.00178
    9000 0.00163
    """
    assert nb_frames
    assert batch_size

    lr = (0.01231 / (math.pow(2.0, (2.0 * math.log10(float(nb_frames)))))) * float(batch_size)

    return lr


@register_lr("primus")
def primus_learning_rate(nb_frames: int, frame_shape: tuple, batch_size: int) -> float:
    """
    nb  lr (bs = 32)
    600 0.00130
    1800 0.00110
    3000 0.001039
    4200 0.000997
    5400 0.000968
    6600 0.000946
    7800 0.000928
    9000 0.000913
    """
    assert nb_frames
    assert batch_size

    lr = (0.000333 / math.log(float(nb_frames))) * float(batch_size)

    return lr


@register_lr("starscream")
def starscream_learning_rate(
    nb_frames: int, frame_shape: tuple, batch_size: int, mask_path: str, factor: float = 1.0
) -> float:
    assert batch_size
    assert mask_path
    assert factor > 0
    assert os.path.exists(mask_path)

    # Imply at least a mask based on saliency and sobelyst masks
    mask = torch.load(mask_path).float()

    assert nb_frames == mask.shape[0], "Wrong mask number of frame"
    assert frame_shape == mask.shape[1:], "Wrong mask output shape"

    avg = torch.mean(mask).item()
    cst = 2.576875e-5

    # If blured mask, specify a factor = 0.5
    lr = ((cst * factor) / (avg + 1e-8)) * float(batch_size)

    return lr
