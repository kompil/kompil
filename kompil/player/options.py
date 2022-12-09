import torch


class PlayerTargetOptions:
    def __init__(self, device: torch.device, dtype: torch.dtype, resolution: str):
        self.device = device
        self.dtype = dtype
        self.resolution = resolution
