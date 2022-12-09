import torch

from typing import Union, Tuple

from torch.utils.data import IterableDataset

from kompil.utils.colorspace import yuv420_to_yuv420n
from kompil.utils.video import display_frame
from .base import register_loss, Loss
from .fct import rectified_yuv420_chrominance, importance_factor

TEXT1 = "Calculating pixels temporal occurences"


def _bsum(t: torch.Tensor):
    nonbdim = tuple(range(1, len(t.shape)))
    return torch.sum(t, dim=nonbdim)


def _pixel_wise_entropy(dataset: IterableDataset, verbose: bool = False):
    first_frame = dataset[0][1]
    frame_shape = dataset[0][1].shape
    nb_frame = len(dataset)
    colorspace = dataset.video_loader.colorspace
    assert dataset.video_loader.colorspace == "yuv420", "Only yuv420 supported"

    # Count pixels occurences
    if verbose:
        print(TEXT1, end="\r")
    histogram = torch.zeros(256, *frame_shape, dtype=torch.uint8, device=first_frame.device)
    for i in range(nb_frame):
        if verbose:
            print(f"{TEXT1}: {100.0 * i/(len(dataset) - 1):0.2f}%", end="\r")

        # Increment counter
        _, ref = dataset[i]
        normalized = yuv420_to_yuv420n(ref)
        discretized = (normalized * 255).long()

        for x in range(256):
            histogram[x] += discretized == x

    if verbose:
        print()

    # Smooth the histogram with gaussian kernel
    smooth_histogram = histogram.view(1, 1, 256, -1).float()
    weight = torch.FloatTensor(
        [[[[0.006], [0.061], [0.242], [0.383], [0.242], [0.061], [0.006]]]]
    ).to(smooth_histogram.device)
    smooth_histogram = torch.nn.functional.conv2d(smooth_histogram, weight, padding=(3, 0))
    smooth_histogram = smooth_histogram.int().view(256, *frame_shape).float()

    # Calculate entropy
    histo_sum = torch.sum(smooth_histogram, dim=(0,))
    histo_log2 = torch.log2(smooth_histogram.float() / histo_sum)
    histo_log2 = torch.nan_to_num(histo_log2, nan=0, posinf=0, neginf=0)
    entropy = -torch.sum(smooth_histogram * histo_log2, dim=(0,))

    # Normalize entropy to [0; 1]
    entropy_range = entropy.max() - entropy.min()
    entropy = (entropy - entropy.min()) / entropy_range

    # Return entropy
    return entropy


@register_loss("entropy")
class EntropyLoss(Loss):
    def __init__(self):
        super().__init__()
        self.__entropy = None
        self.__color_power = None
        self.__emin = None
        self.__emax = None
        self.__mult = None

    def prepare(
        self,
        dataset: IterableDataset,
        color_power: float = 3.0,
        emin: float = 1.0,
        emax: float = 10.0,
        mult: float = 32.0,
        display: bool = False,
    ):
        self.__color_power = color_power
        self.__emin = emin
        self.__emax = emax
        self.__mult = mult

        with torch.no_grad():
            self.__entropy = _pixel_wise_entropy(dataset, verbose=True)

        # Show entropy
        if not display:
            return
        y = torch.nn.functional.pixel_shuffle(self.__entropy[0:4].unsqueeze(0), 2).squeeze(0)
        display_frame(torch.cat([y, y, y]), "Y", wait_time=0)

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor, context: dict):
        assert pred.shape[-3] == 6 and ref.shape[-3] == 6, "Only 420 colorspaces supported"

        # Isolate channels
        pred_y = pred[..., 0:4, :, :]
        pred_u = pred[..., 4:5, :, :]
        pred_v = pred[..., 5:6, :, :]

        ref_y = ref[..., 0:4, :, :]
        ref_u = ref[..., 4:5, :, :]
        ref_v = ref[..., 5:6, :, :]

        entropy_y = self.__entropy[0:4, :, :]
        entropy_u = self.__entropy[4:5, :, :]
        entropy_v = self.__entropy[5:6, :, :]

        # Calculate error
        y_err = torch.abs(pred_y - ref_y)
        u_err = torch.abs(pred_u - ref_u)
        v_err = torch.abs(pred_v - ref_v)

        # Add entropy
        y_err = y_err * (entropy_y * (self.__emax - self.__emin) + self.__emin)
        u_err = u_err * (entropy_u * (self.__emax - self.__emin) + self.__emin)
        v_err = v_err * (entropy_v * (self.__emax - self.__emin) + self.__emin)

        # Calculate power individualy
        y_err = torch.pow(y_err * self.__mult, 2)
        u_err = torch.pow(u_err * self.__mult, self.__color_power)
        v_err = torch.pow(v_err * self.__mult, self.__color_power)

        # Euclidian on result, batch per batch
        epsilon = 1e-6

        batch_eucl = torch.sqrt(_bsum(y_err) + _bsum(u_err) + _bsum(v_err) + epsilon)

        # Return mean so the value does not depend on the batch size
        return batch_eucl.mean()


@register_loss("entropy2")
class Entropy2Loss(Loss):
    """
    Loss for yuv420 images that combines the following techniques:
    - chrominance correction: adds a non-linear function to the chrominance part of the error.
    - euclidian loss: gives an overall objective targeta and ensure the learning direction.
    - entropy: multiply error using pixelwise time-independant entropy.
    - mask: gives frame-per-frame control for specificities of the target video.
    """

    def __init__(self):
        super().__init__()
        self.__entropy = None

    def prepare(
        self,
        dataset: IterableDataset,
        entropy_factor: float = 10.0,
        color_correction: Tuple[float, float, float, float, float] = (1.0, 1.8, 0.2, 0.0555, 0.5),
        mask_path: Union[str, None] = None,
        mask_factor: float = 4.0,
    ):
        self.__color_correction = color_correction

        # Prepare entropy
        with torch.no_grad():
            self.__entropy = _pixel_wise_entropy(dataset, verbose=True)
            self.__entropy = importance_factor(self.__entropy, entropy_factor)

        # Build mask
        self.__mask = None
        if mask_path is None:
            return
        self.__mask = importance_factor(torch.load(mask_path), mask_factor)

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor, context: dict):
        assert pred.shape[-3] == 6 and ref.shape[-3] == 6, "Only 420 colorspaces supported"

        # Get from context
        frames_idx = context["frames_idx"]

        # Some constants
        epsilon = 1e-6
        nonbdim = tuple(range(1, len(pred.shape)))

        # Prepare mask
        if self.__mask is not None:
            self.__mask = self.__mask.to(pred.device).to(pred.dtype)
            mask = self.__mask[frames_idx].view_as(pred)

        # Calculate error
        err = torch.abs(pred - ref)

        # Rectify color
        err = rectified_yuv420_chrominance(err, *self.__color_correction)

        # Euclidian
        mse = torch.square(err)

        # Apply entropy
        mse = mse * self.__entropy

        # Apply mask on mse
        if self.__mask is not None:
            mse = mse * mask

        # Euclidian batch per batch
        batch_euclidian = torch.sqrt(torch.sum(mse, dim=nonbdim) + epsilon)

        # Return mean so the value does not depend on the batch size
        return batch_euclidian.mean()
