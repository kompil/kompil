import torch

from torch.utils.data import IterableDataset

from .base import register_loss, Loss


def _bsum(t: torch.Tensor):
    nonbdim = tuple(range(1, len(t.shape)))
    return torch.sum(t, dim=nonbdim)


@register_loss("mask")
class MaskLoss(Loss):
    def __init__(self):
        super().__init__()
        self.__mask = None

    def prepare(
        self,
        dataset: IterableDataset,
        mask_path: str,
        factor: float = 10.0,
        lumpow: float = 2.0,
        chrpow: float = 2.0,
        mul: float = 1.0,
    ):
        self.__lumpow = lumpow
        self.__chrpow = chrpow
        self.__mul = mul
        # Build mask
        mask = torch.load(mask_path)
        a = 2 / (2 + factor)
        b = 2 * factor / (factor + 2)
        self.__mask = b * mask + a

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor, context: dict):
        assert pred.shape[-3] == 6 and ref.shape[-3] == 6, "Only 420 colorspaces supported"

        # Get from context
        frames_idx = context["frames_idx"]

        # Calculate error
        err = torch.abs(pred - ref)

        # Mask application
        self.__mask = self.__mask.to(pred.device).to(pred.dtype)
        err = err * self.__mask[frames_idx]

        # Isolate luminance and chrominance
        err_y = err[..., 0:4, :, :]
        err_c = err[..., 4:6, :, :]

        # Apply the color weight correction
        err_y = torch.pow(err_y * self.__mul, self.__lumpow)
        err_c = torch.pow(err_c * self.__mul, self.__chrpow)

        # Euclidian on result, batch per batch
        epsilon = 1e-6

        batch_eucl = torch.sqrt(_bsum(err_y) + _bsum(err_c) + epsilon)

        # Return mean so the value does not depend on the batch size
        return batch_eucl.mean()
