import torch
import torchprof

from kompil.data.timeline import create_timeline
from kompil.nn.models.model import model_load


def perf(model_file: str):
    device = torch.device("cuda")
    # Load model
    model = model_load(model_file).to(device)
    nb_frames = model.nb_frames
    timeline = create_timeline(nb_frames, device=device)

    with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
        for frame_id in range(min(nb_frames, 100)):
            time_vec = timeline[frame_id]
            x = time_vec.unsqueeze(0)
            _ = model.forward_rgb8(x)[0]

    print(prof)
