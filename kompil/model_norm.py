from kompil.nn.models.model import model_load, model_save, VideoNet
from kompil.quant.utils.norm import normalize_model


def model_norm(input: str, output: str, cpu: bool):

    model: VideoNet = model_load(input)
    if not cpu:
        model = model.cuda()

    model = normalize_model(model)

    model_save(model, output)
