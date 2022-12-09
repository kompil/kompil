import os
import torch

from kompil.nn.models.model import model_load_from_data
from kompil.quant.factory import factory


def quantize(src_path: str, dst_path: str, method: str):
    assert os.path.exists(src_path)

    data = torch.load(src_path)
    model = model_load_from_data(data)

    quantizer = factory().get(method)()

    qmodel = quantizer.quantize(model)

    new_data = {
        "quantization": {
            "method": method,
            "backend": torch.backends.quantized.engine,
            "version": quantizer.version,
        },
        "model_state_dict": qmodel.state_dict(),
        "model_meta_dict": qmodel.to_meta_dict(),
    }

    data.update(new_data)

    torch.save(data, dst_path)
