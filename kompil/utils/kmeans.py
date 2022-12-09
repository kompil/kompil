import torch
import numpy as np

from tqdm import tqdm


def kmeans(x, k, epochs=100, tol=1e-5):
    nb_sample, dim = x.shape  # Number of samples, dimension of the ambient space

    tqdm_meter = tqdm(desc="[running kmeans]")
    indices = np.random.choice(nb_sample, k, replace=False)
    initial_state_c = x[indices]

    c = initial_state_c  # Simplistic initialization for the centroids
    x_i = torch.tensor(x.view(nb_sample, 1, dim), device=x.device)  # (N, 1, D) samples
    c_j = torch.tensor(c.view(1, k, dim), device=x.device)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(epochs):
        # E step: assign points to the closest cluster
        # d_ij = (torch.abs(x_i - c_j)).sum(-1)  # (N, K) l1 distances
        with torch.no_grad():
            d_ij = torch.sqrt(torch.square(x_i - c_j).sum(-1))  # (N, K) symbolic squared distances
            cl = d_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average
            # Compute the sum of points per cluster
            c_pre = c.clone()
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, dim), x)

            # Divide by the number of points per cluster
            ncl = torch.bincount(cl, minlength=k).type_as(c).view(k, 1)
            c /= ncl  # in-place division to compute the average

            center_shift = torch.sum(torch.sqrt(torch.sum((c - c_pre) ** 2, dim=1)))

        tqdm_meter.set_postfix(
            iteration=f"{i}", center_shift=f"{center_shift ** 2:0.6f}", tol=f"{tol:0.6f}"
        )
        tqdm_meter.update()

        if center_shift**2 < tol:
            break

    return cl, c
