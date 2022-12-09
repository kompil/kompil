import torch


def compute_pixel_dist(img1: torch.Tensor, img2: torch.Tensor):
    euclidian_dist = torch.sqrt(torch.sum(torch.square(img1 - img2), dim=-3, keepdim=True))

    return euclidian_dist


def compute_adjusted_euclidian(img1: torch.Tensor, img2: torch.Tensor):
    """
    Compute the pixel-adjusted euclidian measure.
    """
    mse = torch.mean((img1 - img2) ** 2)
    return torch.sqrt(mse)
