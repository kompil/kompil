import torch


def create_timeline(nb_frames: int, **kwargs) -> torch.Tensor:
    """
    Create a tensor of size (N, 1) containing the all timeline.
    N is the number of possible inputs (frames).
    """
    return torch.arange(0, nb_frames, dtype=torch.float, **kwargs).view(nb_frames, 1)
