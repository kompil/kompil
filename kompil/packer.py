import os
import torch

from kompil.packers import pack_model_data, unpack_model_data


def packer_pack(src_path: str, dst_path: str, packer: str):
    assert os.path.exists(src_path)

    data = torch.load(src_path, map_location="cpu")
    data_packed = pack_model_data(data, packer)

    torch.save(data_packed, dst_path)


def packer_unpack(src_path: str, dst_path: str):
    assert os.path.exists(src_path)

    data_packed = torch.load(src_path, map_location="cpu")
    data = unpack_model_data(data_packed)

    torch.save(data, dst_path)
