import os
import torch

from kompil.nn.models.model import model_load_from_data
from kompil.quant.quantizable import make_quantizable_model
from kompil.quant.mapping import get_blacklist_mapping


def make_quantizable(src_path: str, dst_path: str, blacklist: str):
    assert os.path.exists(src_path)

    data = torch.load(src_path)
    model = model_load_from_data(data)

    blacklist = get_blacklist_mapping().get(blacklist, None)

    qblemodel = make_quantizable_model(model, blacklist)

    new_data = {
        "model_state_dict": qblemodel.state_dict(),
        "model_meta_dict": qblemodel.to_meta_dict(),
    }

    data.update(new_data)

    torch.save(data, dst_path)
