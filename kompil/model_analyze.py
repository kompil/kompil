import os
import torch

from kompil.utils.numbers import to_scale
from kompil.nn.models.model import VideoNet
from kompil.packers.utils import generic_load


def count_zeros(model: VideoNet):
    counter = 0
    for t_params in model.parameters():
        counter += torch.sum(t_params.data == 0).item()
    return counter


def analyse_model(model_fpath: str):
    model_fpath = os.path.expanduser(model_fpath)
    model: VideoNet = generic_load(model_fpath)

    # Some calculations
    nb_params = model.nb_params
    nb_zero_params = count_zeros(model)
    nb_active_params = nb_params - nb_zero_params
    post_huffman_eq = nb_active_params + int(nb_zero_params / 8)

    # First print the full model
    print(model)

    # Some data about the video
    print("Number of parameters:", to_scale(nb_params))
    print("Number of zeros:", to_scale(nb_zero_params))
    print("Number of active weight (non zero):", to_scale(nb_active_params))
    if nb_params != 0:
        print(f"Purcent of active weight: {nb_active_params/nb_params*100:0.1f}%")
    print("Post-huffman weight count equivalent:", to_scale(post_huffman_eq))
