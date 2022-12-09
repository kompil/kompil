import torch


def rectify_error(
    error: torch.Tensor, a: float, b: float, c: float, f: float, g: float
) -> torch.Tensor:
    """
    Apply a continuous function that decrease the importance of close to 0 values, then increase it
    and ultimately behave like a "y = x + cste" function.

    :param a: Overall multiple. ]0; +inf]
    :param b: Multiple that will decay over time. [0; +inf]
    :param c: Exponent that will decay over time. [0; +inf]
    :param f: Define the decay characteristics of the multiple. [0; 1]
    :param c: Define the decay characteristics of the exponent. [0; 1]
    """
    factor = (error + b) / (error + f * b)
    power = (error + c) / (error + g * c)
    return a * torch.pow(error * factor, power)


def rectified_yuv420_chrominance(
    error: torch.Tensor, a: float, b: float, c: float, f: float, g: float
):
    """
    Rectify the chrominance-related error to give it more importance.
    """
    rectified_error = torch.empty_like(error)
    rectified_error[..., 0:4, :, :] = error[..., 0:4, :, :]
    rectified_error[..., 4:6, :, :] = rectify_error(error[..., 4:6, :, :], a, b, c, f, g)
    return rectified_error


def rectified_chrominance_butterfly(error: torch.Tensor, factor: float, upow: float, vpow: float):
    """
    Rectify the chrominance-related error to give it more importance.
    """
    rectified_error = torch.empty_like(error)

    rectified_error[..., 0:4, :, :] = error[..., 0:4, :, :] * factor
    rectified_error[..., 4:5, :, :] = torch.pow(error[..., 4:5, :, :] * factor, upow)
    rectified_error[..., 5:6, :, :] = torch.pow(error[..., 5:6, :, :] * factor, vpow)

    return rectified_error


def importance_factor(data: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Apply a linear transformation to a 0.0 to 1.0 tensor such that the value at the end has a factor
    between the lowest value and the highest.
    For instance, with a factor 10.0, 1.0 stay 1.0 and 0.0 become 0.1.
    """
    a = 2 / (2 + factor)
    b = 2 * factor / (factor + 2)
    return b * data + a
