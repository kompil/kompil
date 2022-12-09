import math
import torch
import kornia

from typing import Union, Tuple

from torch.utils.data import IterableDataset
from .base import register_loss, Loss
from .fct import rectified_yuv420_chrominance, rectified_chrominance_butterfly


@register_loss("iron")
class IronLoss(Loss):
    """
    Loss for yuv420 images that combines the following techniques:
    - chrominance correction: adds a non-linear function to the chrominance part of the error.
    - euclidian loss: gives an overall objective targeta and ensure the learning direction.
    - ssim loss: increase details and avoid some noise.
    - mask multiple: gives frame-per-frame control for specificities of the target video.
    """

    def prepare(
        self,
        dataset: IterableDataset,
        color_correction: Tuple[float, float, float] = (32.0, 1.5, 1.5),
        ssim_window: Union[int, None] = 7,
        mask_path: Union[str, None] = None,
        mask_factor: float = 4.0,
    ):
        self.__color_correction = color_correction
        self.__ssim_window = ssim_window
        self.__ssim_rescaling = math.pow(
            color_correction[0], (color_correction[1] + color_correction[2]) / 2
        )
        # Build mask
        self.__mask = None
        if mask_path is None:
            return
        mask = torch.load(mask_path)
        a = 2 / (2 + mask_factor)
        b = 2 * mask_factor / (mask_factor + 2)
        self.__mask = b * mask + a

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor, context: dict):
        assert pred.shape[-3] == 6 and ref.shape[-3] == 6, "Only 420 colorspaces supported"

        # Some constants
        epsilon = 1e-6
        batches = pred.shape[0]
        nonbdim = tuple(range(1, len(pred.shape)))

        # Prepare mask
        if self.__mask is not None:
            self.__mask = self.__mask.to(pred.device).to(pred.dtype)
            frames_idx = context["frames_idx"]
            mask = self.__mask[frames_idx].view_as(pred)

        # Calculate error
        err = torch.abs(pred - ref)

        # rectify color
        err = rectified_chrominance_butterfly(err, *self.__color_correction)

        # Euclidian
        mse = torch.square(err)

        # Apply mask on mse
        if self.__mask is not None:
            mse = mse * mask

        # Euclidian batch per batch
        batch_euclidian = torch.sqrt(torch.sum(mse, dim=nonbdim) + epsilon)

        # Allow disabling the SSIM part of the loss
        if self.__ssim_window is None:
            return batch_euclidian.mean()

        # SSIM
        ssim = kornia.losses.ssim_loss(pred, ref, window_size=self.__ssim_window, reduction="none")

        # Scale SSIM loss to match color correction
        ssim *= self.__ssim_rescaling

        # Apply mask on SSIM
        if self.__mask is not None:
            ssim = ssim * mask

        # SSIM batch per batch
        batch_ssim = torch.sqrt(torch.sum(ssim, dim=nonbdim))

        # Multiply batch per batch and mean the total
        dot_product = torch.dot(batch_euclidian.t(), batch_ssim)
        return dot_product / batches


@register_loss("iron2")
class Iron2Loss(Loss):
    """
    Loss for yuv420 images that combines the following techniques:
    - chrominance correction: adds a non-linear function to the chrominance part of the error.
    - euclidian loss: gives an overall objective targeta and ensure the learning direction.
    - ssim loss: increase details and avoid some noise.
    - mask multiple: gives frame-per-frame control for specificities of the target video.
    """

    def prepare(
        self,
        dataset: IterableDataset,
        color_correction: Tuple[float, float, float, float, float] = (1.0, 1.8, 0.2, 0.0555, 0.5),
        ssim_window: Union[int, None] = 7,
        mask_path: Union[str, None] = None,
        mask_factor: float = 4.0,
    ):
        self.__color_correction = color_correction
        self.__ssim_window = ssim_window
        # Build mask
        self.__mask = None
        if mask_path is None:
            return
        mask = torch.load(mask_path)
        a = 2 / (2 + mask_factor)
        b = 2 * mask_factor / (mask_factor + 2)
        self.__mask = b * mask + a

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor, context: dict):
        assert pred.shape[-3] == 6 and ref.shape[-3] == 6, "Only 420 colorspaces supported"

        # Some constants
        epsilon = 1e-6
        batches = pred.shape[0]
        nonbdim = tuple(range(1, len(pred.shape)))

        # Prepare mask
        if self.__mask is not None:
            self.__mask = self.__mask.to(pred.device).to(pred.dtype)
            frames_idx = context["frames_idx"]
            mask = self.__mask[frames_idx].view_as(pred)

        # Calculate error
        err = torch.abs(pred - ref)

        # rectify color
        err = rectified_yuv420_chrominance(err, *self.__color_correction)

        # Euclidian
        mse = torch.square(err)

        # Apply mask on mse
        if self.__mask is not None:
            mse = mse * mask

        # Euclidian batch per batch
        batch_euclidian = torch.sqrt(torch.sum(mse, dim=nonbdim) + epsilon)

        # Allow disabling the SSIM part of the loss
        if self.__ssim_window is None:
            return batch_euclidian.mean()

        # SSIM
        ssim = kornia.losses.ssim_loss(pred, ref, window_size=self.__ssim_window, reduction="none")

        # Apply mask on SSIM
        if self.__mask is not None:
            ssim = ssim * mask

        # SSIM batch per batch
        batch_ssim = torch.sqrt(torch.sum(ssim, dim=nonbdim))

        # Multiply batch per batch and mean the total
        dot_product = torch.dot(batch_euclidian.t(), batch_ssim)
        return dot_product / batches
