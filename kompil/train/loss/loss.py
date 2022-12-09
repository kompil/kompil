import math
import torch
import kornia

from kompil.utils.colorspace import colorspace_444_to_yuv420
from .base import register_loss

MAE_LOSS = "l1"
MSE_LOSS = "l2"
HUBER_LOSS = "huber"
SSIM_LOSS = "ssim"
WED_LOSS = "wed"
EUCLIDIAN_LOSS = "euclidian"
PSNR_LOSS = "psnr"
SIMPLIFIED_ADAPTIVE_LOSS = "simplified_adaptive"
ADAPTIVE_LOSS = "adaptive"
LOG_MSE = "log_mse"
LOG_MSE_BOUND = "log_mse_bound"
EXS_LOSS = "exs"
EPS_LOSS = "eps"
EPF_LOSS = "epf"
BATCH_SSIM_LOSS = "batch_ssim"
BATCH_EUCLIDIAN_LOSS = "batch_euclidian"
BATCH_EPS_LOSS = "batch_eps"
BATCH_EXS_LOSS = "batch_exs"


@register_loss(PSNR_LOSS)
def psnr_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    mse = torch.mean((y_pred - y_ref) ** 2)
    return -20 * torch.log10(1.0 / torch.sqrt(mse))


@register_loss(MAE_LOSS)
def l1_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    # https://en.wikipedia.org/wiki/Mean_absolute_error
    return torch.nn.functional.l1_loss(y_pred, y_ref)


@register_loss(MSE_LOSS)
def l2_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    # https://en.wikipedia.org/wiki/Mean_squared_error
    return torch.nn.functional.mse_loss(y_pred, y_ref)


@register_loss(HUBER_LOSS)
def huber_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    # https://en.wikipedia.org/wiki/Huber_loss
    return torch.nn.functional.smooth_l1_loss(y_pred, y_ref)


@register_loss(SSIM_LOSS)
def ssim_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict, reduction="sum"):
    # https://en.wikipedia.org/wiki/Structural_similarity
    return kornia.losses.ssim_loss(y_pred, y_ref, window_size=7, reduction=reduction)


@register_loss(WED_LOSS)
def wed_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    # https://hal.archives-ouvertes.fr/hal-01223730/document
    sq = (y_pred - y_ref) ** 2
    g = torch.exp(-(sq / 2)) * sq
    return g.sum()


@register_loss(EUCLIDIAN_LOSS)
def euclidian_dist_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    # https://en.wikipedia.org/wiki/Euclidean_distance
    return torch.sqrt(torch.sum(torch.square(y_pred - y_ref)))


@register_loss(EXS_LOSS)
def exs_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    eucl_val = euclidian_dist_loss(y_pred, y_ref)
    ssim_val = ssim_loss(y_pred, y_ref, reduction="mean")
    return eucl_val * ssim_val


@register_loss(EPS_LOSS)
def eps_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    eucl_val = euclidian_dist_loss(y_pred, y_ref, context)
    ssim_val = ssim_loss(y_pred, y_ref, reduction="sum")
    sqrt_ssim = torch.sqrt(ssim_val)
    return eucl_val * 3.45 + sqrt_ssim


@register_loss(BATCH_EUCLIDIAN_LOSS)
def batch_euclidian_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict, batchred="sum"):
    """
    Apply euclidian distances on each batch and then add it (or mean it) together.
    Euclidian: https://en.wikipedia.org/wiki/Euclidean_distance

    :param batchred: "sum" or "mean"
    """
    nonbdim = tuple(range(1, len(y_pred.shape)))
    eucl_by_img = torch.sqrt(torch.sum(torch.square(y_pred - y_ref), dim=nonbdim))

    if batchred == "mean":
        return torch.mean(eucl_by_img)
    elif batchred == "sum":
        return torch.sum(eucl_by_img)
    else:
        raise RuntimeError(f"reduction {batchred} not known")


@register_loss(BATCH_SSIM_LOSS)
def batch_sqrt_ssim_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict, reduction="sum"):
    """
    Apply ssim loss on each batch and then add it (or mean it) together.
    SSIM: https://en.wikipedia.org/wiki/Structural_similarity

    :param batchred: "sum" or "mean"
    """
    nonbdim = tuple(range(1, len(y_pred.shape)))

    sqrt_ssim = torch.sqrt(
        torch.sum(
            kornia.losses.ssim_loss(y_pred, y_ref, reduction="none", window_size=7), dim=nonbdim
        )
    )

    if reduction == "mean":
        return torch.mean(sqrt_ssim)
    elif reduction == "sum":
        return torch.sum(sqrt_ssim)
    else:
        raise RuntimeError(f"reduction {reduction} not known")


@register_loss(BATCH_EPS_LOSS)
def batch_eps_loss(
    y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict, reduction="sum", ratio=3.45
):
    """
    Apply ssim loss and euclidian on each batch and then add it together based on a specidied ratio.

    :param batchred: "sum" or "mean"
    :param ratio: euclidian * ratio + sqrt(ssim)
    """
    nonbdim = tuple(range(1, len(y_pred.shape)))
    epsilon = 1e-6  # Avoid sqrt(0)

    euclidian = torch.sqrt(torch.sum(torch.square(y_pred - y_ref), dim=nonbdim) + epsilon)
    sqrt_ssim = torch.sqrt(
        torch.sum(
            kornia.losses.ssim_loss(y_pred, y_ref, reduction="none", window_size=7), dim=nonbdim
        )
        + epsilon
    )

    value = euclidian * ratio + sqrt_ssim

    if reduction == "mean":
        value /= y_pred.shape[0]

    return value


@register_loss(BATCH_EXS_LOSS)
def batch_exs_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict, reduction="sum"):
    nonbdim = tuple(range(1, len(y_pred.shape)))
    epsilon = 1e-6  # Avoid sqrt(0)

    euclidian = torch.sqrt(torch.sum(torch.square(y_pred - y_ref), dim=nonbdim) + epsilon)
    sqrt_ssim = torch.sqrt(
        torch.sum(
            kornia.losses.ssim_loss(y_pred, y_ref, reduction="none", window_size=7), dim=nonbdim
        )
        + epsilon
    )

    # The dot product multiply element-wise and then add everything together.
    dot_product = torch.dot(euclidian.t(), sqrt_ssim)

    if reduction == "mean":
        dot_product /= y_pred.shape[0]

    return dot_product


@register_loss(EPF_LOSS)
def epf_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    nonbdim = tuple(range(1, len(y_pred.shape)))

    eucl = torch.sum(torch.sqrt(torch.sum(torch.square(y_pred - y_ref), dim=nonbdim))) * 8

    fft_pred = torch.view_as_real(torch.fft.rfftn(y_pred, dim=nonbdim))
    fft_ref = torch.view_as_real(torch.fft.rfftn(y_ref, dim=nonbdim))
    fft = torch.sqrt(torch.sum(torch.abs(fft_pred - fft_ref)))

    return eucl + fft


@register_loss(ADAPTIVE_LOSS)
def adaptive_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict, a, c):
    # https://arxiv.org/abs/1701.03077
    # Must be declared as Variable(torch.ones(output_size).cuda(), requires_grad=True)
    a = 2 * torch.sigmoid(a)
    c = torch.nn.functional.softplus(c) + 1e-8

    return (
        torch.true_divide(torch.abs(a - 2), a)
        * (
            torch.pow(
                (
                    torch.true_divide(
                        torch.square(torch.true_divide(y_pred - y_ref, c)), torch.abs(a - 2)
                    )
                )
                + 1,
                torch.true_divide(a, 2),
            )
            - 1
        )
    ).mean()


@register_loss(SIMPLIFIED_ADAPTIVE_LOSS)
def simplified_adaptive_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    # https://arxiv.org/abs/1701.03077 where a = 1, c = 0.01
    return (torch.pow(torch.square(torch.true_divide(y_pred - y_ref, 0.01)) + 1, 0.5) - 1).mean()


@register_loss(LOG_MSE)
def log_mse(y_pred: torch.Tensor, y_ref: torch.Tensor):
    mse = torch.sum(torch.square(y_pred - y_ref))
    return torch.log(1 + mse)


@register_loss(LOG_MSE_BOUND)
def log_mse_bound(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict, bound=45.0):
    # Calculate bound based on target pnsr
    a = 1 - 1 / (torch.pow(10, bound / 10.0))
    # Log MSE
    mse = torch.sum(torch.square(y_pred - y_ref))
    return torch.log(a + mse).clamp(0.0, math.inf)


@register_loss("batch_yuv420")
def batch_yuv420(pred: torch.Tensor, ref: torch.Tensor, context: dict, alpha=0.005):
    nonbdim = tuple(range(1, len(pred.shape)))
    epsilon = 1e-6  # Avoid sqrt(0)

    yuv420_pred = colorspace_444_to_yuv420(kornia.color.yuv.rgb_to_yuv(pred))
    yuv420_ref = colorspace_444_to_yuv420(kornia.color.yuv.rgb_to_yuv(ref))

    euclidian_gross = torch.sqrt(
        torch.sum(torch.square(yuv420_pred - yuv420_ref), dim=nonbdim) + epsilon
    )
    euclidian_details = torch.sqrt(torch.sum(torch.square(pred - ref), dim=nonbdim) + epsilon)
    return (euclidian_gross + alpha * euclidian_details).mean()


@register_loss("neon")
def neon_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    nonbdim = tuple(range(1, len(y_pred.shape)))
    epsilon = 1e-6  # Avoid sqrt(0)

    y_y_pred = y_pred[:, 0:4, :, :]
    y_y_ref = y_ref[:, 0:4, :, :]
    u_y_pred = y_pred[:, 4:5, :, :]
    u_y_ref = y_ref[:, 4:5, :, :]
    v_y_pred = y_pred[:, 5:6, :, :]
    v_y_ref = y_ref[:, 5:6, :, :]

    # Luminance component
    y_diff = torch.square(y_y_pred - y_y_ref)
    y_euclidian = torch.sqrt(torch.sum(y_diff, dim=nonbdim) + epsilon)
    luminance = torch.sum(y_euclidian)

    # Progressive chromatic euclidian
    u_euclidian = torch.sqrt(
        torch.sum(torch.square(u_y_pred - u_y_ref), dim=nonbdim) + epsilon
    ).sum()
    v_euclidian = torch.sqrt(
        torch.sum(torch.square(v_y_pred - v_y_ref), dim=nonbdim) + epsilon
    ).sum()
    u_factor = (1 - y_diff.mean() * 1000).clamp(0, 1) * 0.50
    v_factor = u_factor * 0.25
    progressive_chromatic = u_euclidian * u_factor + v_euclidian * v_factor

    # Neon loss
    loss = luminance + progressive_chromatic

    return loss


@register_loss("butterfly")
def butterfly_loss(pred: torch.Tensor, ref: torch.Tensor, context: dict, upow=2, vpow=2, factor=64):
    """
    Butterly is basically like a batch euclidian loss but trying to put more pressure on the
    color outliers.
    The way it does that is by increasing the exponent on the U and V errors. For instance 4 instead
    of 2. There is a factor added to it in order to compensate the value reduction, introduced by
    the higher power, when error is to low.

    This loss only accept the 420 colorspaces where the last 2 channels defines the color.
    """
    assert pred.shape[-3] == 6 and ref.shape[-3] == 6, "Only 420 colorspaces supported"

    # Isolate channels
    pred_y = pred[..., 0:4, :, :]
    pred_u = pred[..., 4:5, :, :]
    pred_v = pred[..., 5:6, :, :]

    ref_y = ref[..., 0:4, :, :]
    ref_u = ref[..., 4:5, :, :]
    ref_v = ref[..., 5:6, :, :]

    # Calculate error
    y_err = torch.abs(pred_y - ref_y)
    u_err = torch.abs(pred_u - ref_u)
    v_err = torch.abs(pred_v - ref_v)

    # Calculate power individualy
    y_err = torch.pow(y_err * factor, 2)
    u_err = torch.pow(u_err * factor, 2 * upow)
    v_err = torch.pow(v_err * factor, 2 * vpow)

    # Euclidian on result, batch per batch
    epsilon = 1e-6

    def _bsum(t: torch.Tensor):
        nonbdim = tuple(range(1, len(t.shape)))
        return torch.sum(t, dim=nonbdim)

    batch_eucl = torch.sqrt(_bsum(y_err) + _bsum(u_err) + _bsum(v_err) + epsilon)

    # Return mean so the value does not depend on the batch size
    return batch_eucl.mean()


@register_loss("neoptera")
def neoptera_loss(
    pred: torch.Tensor,
    ref: torch.Tensor,
    context: dict,
    upow=2,
    vpow=2,
    maxpow=6,
    factor=48,
):
    """
    Neoptera is Butterfly-based with 3 modifications :
    - A major malus for outliers
    - A stronger gradient on low differences
    - Less prone to very low / very high values (~ which could be unstable)

    This loss only accept the 420 colorspaces where the last 2 channels defines the color.
    """
    assert pred.shape[-3] == 6 and ref.shape[-3] == 6, "Only 420 colorspaces supported"

    # Isolate channels
    pred_y = pred[..., 0:4, :, :]
    pred_u = pred[..., 4:5, :, :]
    pred_v = pred[..., 5:6, :, :]

    ref_y = ref[..., 0:4, :, :]
    ref_u = ref[..., 4:5, :, :]
    ref_v = ref[..., 5:6, :, :]

    # Diff
    y_diff = pred_y - ref_y
    u_diff = pred_u - ref_u
    v_diff = pred_v - ref_v

    # Calculate error
    y_abs = torch.abs(y_diff)
    u_abs = torch.abs(u_diff)
    v_abs = torch.abs(v_diff)

    # Square malus for outliers
    y_sq = torch.square(y_diff)
    u_sq = torch.square(u_diff)
    v_sq = torch.square(v_diff)

    # Calculate power individualy, with additional malus for high l2 difference
    # Add clamping to avoid to small/big values
    epsilon = 1e-6
    y_pow = torch.clamp(torch.pow(y_abs * factor, torch.clamp(y_sq + 2, 0, maxpow)), epsilon)
    u_pow = torch.clamp(torch.pow(u_abs * factor, torch.clamp(u_sq + 2 * upow, 0, maxpow)), epsilon)
    v_pow = torch.clamp(torch.pow(v_abs * factor, torch.clamp(v_sq + 2 * vpow, 0, maxpow)), epsilon)

    # Add malus for low values
    y_err = y_pow + torch.sqrt(y_pow + epsilon)
    u_err = u_pow + torch.sqrt(u_pow + epsilon)
    v_err = v_pow + torch.sqrt(v_pow + epsilon)

    # Euclidian on result, batch per batch
    def _bsum(t: torch.Tensor):
        nonbdim = tuple(range(1, len(t.shape)))
        return torch.sum(t, dim=nonbdim)

    batch_eucl = torch.sqrt(_bsum(y_err) + _bsum(u_err) + _bsum(v_err) + epsilon)

    # Return mean so the value does not depend on the batch size
    return batch_eucl.mean()
