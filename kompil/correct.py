import os
import torch

from kompil.nn.models.model import model_load, model_load_from_data
from kompil.corr.factory import factory


def correct(o_path: str, q_path: str, dst_path: str, method: str):
    assert os.path.exists(o_path)
    assert os.path.exists(q_path)

    data = torch.load(q_path)

    assert data.get("quantization", None), "Only quantized model can be corrected"

    qmodel = model_load_from_data(data)
    omodel = model_load(o_path)

    corrector = factory().get(method)()

    cmodel = corrector.correct(omodel, qmodel)

    new_data = {
        "correction": {
            "method": method,
            "version": corrector.version,
        },
        "model_state_dict": cmodel.state_dict(),
        "model_meta_dict": cmodel.to_meta_dict(),
    }

    data.update(new_data)

    torch.save(data, dst_path)
