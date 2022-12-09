import torch
import kornia

from torch.nn.quantized import QFunctional

COLORSPACE_LIST = ["rgb8", "yuv420", "ycbcr420", "ycbcr420shift", "ycocg420"]
COLORSPACE_420_LIST = ["yuv420", "ycbcr420", "ycbcr420shift", "ycocg420"]

DEFAULT_420_TO_444_MODE = "nearest"
DEFAULT_444_TO_420_MODE = "bilinear"


def quant_yuv444_to_rgb(x: torch.Tensor, quant_ops: QFunctional = None) -> torch.Tensor:
    r"""Convert an YUV image to RGB.
    The image data is assumed to be in the range of (0, 1).
    Args:
        image (torch.Tensor): YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.
    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.
    """
    assert len(x.shape) >= 3 and x.shape[-3] == 3

    if quant_ops is None:
        quant_ops = QFunctional()

    y: torch.Tensor = x[..., 0, :, :]
    u: torch.Tensor = x[..., 1, :, :]
    v: torch.Tensor = x[..., 2, :, :]

    r: torch.Tensor = quant_ops.add(y, quant_ops.mul_scalar(v, 1.14))  # coefficient for g is 0
    g: torch.Tensor = quant_ops.add(
        y, quant_ops.add(quant_ops.mul_scalar(u, -0.396), quant_ops.mul_scalar(v, -0.581))
    )
    b: torch.Tensor = quant_ops.add(y, quant_ops.mul_scalar(u, 2.029))  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out


def colorspace_420_to_444(x: torch.Tensor, mode=DEFAULT_420_TO_444_MODE) -> torch.Tensor:
    """
    Transform an image yuv420 6:w/2:h/2 into yuv444 3:w:h.
    The 6 input channels represents the 4 Y and 1 U and 1 V.
    """
    assert len(x.shape) == 4
    assert x.shape[1] == 6

    b, _, h_2, w_2 = x.shape
    h, w = h_2 * 2, w_2 * 2

    y_base = x[:, 0:4, :, :]
    y = torch.nn.functional.pixel_shuffle(y_base, 2)

    u_base = x[:, 4:5, :, :]
    u = torch.nn.functional.interpolate(u_base, (h, w), mode=mode)
    v_base = x[:, 5:6, :, :]
    v = torch.nn.functional.interpolate(v_base, (h, w), mode=mode)

    return torch.cat([y, u, v], dim=1)


def colorspace_444_to_yuv420(x: torch.Tensor, mode=DEFAULT_444_TO_420_MODE) -> torch.Tensor:
    """
    Transform an image yuv444 3:w:h into yuv420 6:w/2:h/2.
    This is a lossy conversion.
    """
    assert len(x.shape) == 4
    assert x.shape[1] == 3

    bs, ic, ih, iw = x.shape

    assert ic == 3
    assert ih % 2 == 0
    assert iw % 2 == 0

    uv_size = (int(ih / 2), int(iw / 2))
    uv = x[:, 1:3, :, :]
    uv = torch.nn.functional.interpolate(uv, size=uv_size, mode=mode, align_corners=True)
    y = x[:, 0:1, :, :]
    y = torch.nn.functional.pixel_unshuffle(y, downscale_factor=2)

    yuv420 = torch.empty(bs, 6, uv_size[0], uv_size[1], device=x.device, dtype=x.dtype)
    yuv420[:, 0:4].copy_(y)
    yuv420[:, 4:6].copy_(uv)

    return yuv420


def ycocg444_to_rgb(x: torch.Tensor) -> torch.Tensor:
    y = x[..., 0, :, :]
    co = x[..., 1, :, :]
    cg = x[..., 2, :, :]

    tmp: torch.Tensor = y - cg
    r: torch.Tensor = tmp + co
    g: torch.Tensor = y + cg
    b: torch.Tensor = tmp - co

    rgb: torch.Tensor = torch.stack([r, g, b], -3)

    return rgb


def rgb_to_ycocg444(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[1] == 3

    r = x[..., 0, :, :]
    g = x[..., 1, :, :]
    b = x[..., 2, :, :]

    y: torch.Tensor = 0.25 * r + 0.5 * g + 0.25 * b
    co: torch.Tensor = 0.5 * r - 0.5 * b
    cg: torch.Tensor = -0.25 * r + 0.5 * g - 0.25 * b

    ycocg: torch.Tensor = torch.stack([y, co, cg], -3)

    return ycocg


def convert_to_colorspace(
    images: torch.Tensor, src: str, dst: str, lossy_allowed: bool = False
) -> torch.Tensor:
    assert len(images.shape) in [3, 4], "Implemented only for 3D and 4D tensors"

    if len(images.shape) == 3:
        return convert_to_colorspace(images.unsqueeze(0), src, dst, lossy_allowed).squeeze(0)

    if src == dst:
        return images

    if src == "rgb8" and dst == "yuv420":
        assert lossy_allowed
        return colorspace_444_to_yuv420(kornia.color.yuv.rgb_to_yuv(images))

    if src == "rgb8" and dst == "ycocg420":
        assert lossy_allowed
        return colorspace_444_to_yuv420(rgb_to_ycocg444(images))

    if src == "ycbcr420" and dst == "yuv420":
        assert lossy_allowed
        return ycbcr420_to_yuv420(images)

    if src == "ycbcr420shift" and dst == "ycbcr420":
        return ycbcr420shift_to_ycbcr420(images)

    if src == "ycbcr420" and dst == "ycbcr420shift":
        return ycbcr420_to_ycbcr420shift(images)

    if dst == "rgb8":
        if src == "ycbcr420shift":
            images = ycbcr420shift_to_ycbcr420(images)
            src = "ycbcr420"

        if src in COLORSPACE_420_LIST:
            images = colorspace_420_to_444(images)

        if src == "yuv420":
            return kornia.color.yuv.yuv_to_rgb(images)

        if src == "ycocg420":
            return ycocg444_to_rgb(images)

        if src == "ycbcr420":
            return ycbcr_to_rgb(images)

    raise RuntimeError(f"unimplemented colorspace transition: {src} to {dst}")


def convert_shape_to_colorspace(shape: torch.Size, src: str, dst: str) -> torch.Size:
    assert len(shape) in [3, 4], "Implemented only for 3D and 4D tensors"

    if src == dst:
        return shape

    if src == "rgb8" and dst in COLORSPACE_420_LIST:
        ic, ih, iw = shape[-3:]
        assert ic == 3 and ih % 2 == 0 and iw % 2 == 0
        return torch.Size([*shape[:-3], 6, ih // 2, iw // 2])

    if src in COLORSPACE_420_LIST and dst == "rgb8":
        ic, ih, iw = shape[-3:]
        assert ic == 6
        return torch.Size([*shape[:-3], 3, ih * 2, iw * 2])

    raise RuntimeError(f"unimplemented colorspace transition: {src} to {dst}")


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    Transform a ycbcr batch into a rgb colorspace. The default standard is BT.601.
    """
    return ycbcr_to_rgb_bt601(ycbcr)


def ycbcr_to_rgb_bt601(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    Transform a ycbcr batch into a rgb colorspace. The standard is BT.601.
    See https://en.wikipedia.org/wiki/YCbCr
    """
    y = ycbcr[..., 0, :, :]
    cb = ycbcr[..., 1, :, :]
    cr = ycbcr[..., 2, :, :]
    rgb = torch.empty_like(ycbcr)
    rgb[..., 0, :, :] = 1.164382813 * y + 1.596027344 * cr - 0.870785156
    rgb[..., 1, :, :] = 1.164382813 * y - 0.391761719 * cb - 0.81296875 * cr + 0.52959375
    rgb[..., 2, :, :] = 1.164382813 * y + 2.017234375 * cb - 1.081390625
    rgb.clip_(0.0, 1.0)
    return rgb


def ycbcr_to_rgb_bt709(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    Transform a ycbcr batch into a rgb colorspace. The standard is BT.709.
    See https://en.wikipedia.org/wiki/YCbCr
    """
    y = ycbcr[..., 0, :, :]
    cb = ycbcr[..., 1, :, :]
    cr = ycbcr[..., 2, :, :]
    rgb = torch.empty_like(ycbcr)
    cb = cb - 0.5
    cr = cr - 0.5
    rgb[..., 0, :, :] = y + 1.5748 * cr
    rgb[..., 1, :, :] = y - 0.1873 * cb - 0.4681 * cr
    rgb[..., 2, :, :] = y + 1.8556 * cb
    rgb.clip_(0.0, 1.0)
    return rgb


def ycbcr_to_rgb_bt2020(ycbcr: torch.Tensor) -> torch.Tensor:
    """
    Transform a ycbcr batch into a rgb colorspace. The standard is BT.2020.
    See https://en.wikipedia.org/wiki/YCbCr
    """
    y = ycbcr[..., 0, :, :]
    cb = ycbcr[..., 1, :, :]
    cr = ycbcr[..., 2, :, :]
    rgb = torch.empty_like(ycbcr)
    cb = cb - 0.5
    cr = cr - 0.5
    rgb[..., 0, :, :] = y + 1.4746 * cr
    rgb[..., 1, :, :] = y - 0.1645531 * cb - 0.571353 * cr
    rgb[..., 2, :, :] = y + 1.8814 * cb
    rgb.clip_(0.0, 1.0)
    return rgb


def ycbcr420_to_ycbcr420shift(x: torch.Tensor) -> torch.Tensor:
    """
    Transform a ycbcr420 batch into a ycbcr420shift colorspace.
    """
    o = torch.empty_like(x)
    o[:, 0:4, :, :] = x[:, 0:4, :, :]
    o[:, 4:6, :, :] = x[:, 4:6, :, :] - 0.5
    return o


def ycbcr420shift_to_ycbcr420(x: torch.Tensor) -> torch.Tensor:
    """
    Transform a ycbcr420shift batch into a ycbcr420shift colorspace.
    """
    o = torch.empty_like(x)
    o[:, 0:4, :, :] = x[:, 0:4, :, :]
    o[:, 4:6, :, :] = x[:, 4:6, :, :] + 0.5
    return o


def ycbcr420_to_yuv420(x: torch.Tensor) -> torch.Tensor:
    """
    Transform a ycbcr420 batch into a yuv420 colorspace.
    """
    rgb = kornia.color.ycbcr.ycbcr_to_rgb(colorspace_420_to_444(x))
    yuv420 = colorspace_444_to_yuv420(kornia.color.yuv.rgb_to_yuv(rgb))
    return yuv420


def yuv420_to_yuv420n(x: torch.Tensor) -> torch.Tensor:
    """
    Transform a yuv420 batch into a yuv420 normalized batch.
    """
    assert len(x.shape) in [3, 4]

    if len(x.shape) == 3:
        return yuv420_to_yuv420n(x.unsqueeze(0)).squeeze(0)

    o = torch.empty_like(x)
    o[:, 0:4, :, :] = x[:, 0:4, :, :]
    o[:, 4:5, :, :] = (x[:, 4:5, :, :] + 0.436) / 0.872
    o[:, 5:6, :, :] = (x[:, 5:6, :, :] + 0.615) / 1.23

    return o


def yuv420n_to_yuv420(x: torch.Tensor) -> torch.Tensor:
    """
    Transform a yuv420 normalized batch into a yuv420 batch.
    """
    assert len(x.shape) in [3, 4]

    if len(x.shape) == 3:
        return yuv420n_to_yuv420(x.unsqueeze(0)).squeeze(0)

    o = torch.empty_like(x)
    o[:, 0:4, :, :] = x[:, 0:4, :, :]
    o[:, 4:5, :, :] = x[:, 4:5, :, :] * 0.872 - 0.436
    o[:, 5:6, :, :] = x[:, 5:6, :, :] * 1.23 - 0.615

    return o
