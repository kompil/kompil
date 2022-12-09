import torch
from kompil.nn.models.model import VideoNet


def generic_load(file_path: str) -> VideoNet:
    from kompil.nn.models.model import model_load_from_data
    from kompil.packers import unpack_model_data

    data = torch.load(file_path, map_location="cpu")

    if "unpacker" in data:
        data = unpack_model_data(data)

    return model_load_from_data(data)
