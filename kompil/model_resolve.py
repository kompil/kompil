import os
from typing import Union
from kompil.nn.models.model import VideoNet, model_load, model_save
from kompil.nn.layers.save_load import load_load_layers


def resolve(model_file: str, target_file: Union[str, None]):
    # Guess target file:
    if target_file is None:
        folder = os.path.dirname(model_file)
        name, ext = os.path.splitext(os.path.basename(model_file))
        target_file = os.path.join(folder, name + ".resolved" + ext)
    # Load model loading the external modules
    model: VideoNet = model_load(model_file)
    load_load_layers(model)
    # Save the file with the loaded modules
    model_save(model, target_file)
