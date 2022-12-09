import numpy
import torch

from PyQt5.QtGui import QImage


def __torch_tensor_to_numpy(tensor: torch.ByteTensor) -> numpy.array:
    assert tensor.dtype == torch.uint8
    tensor = tensor.permute(1, 2, 0)
    return tensor.cpu().numpy()


def __torch_tensor_to_uint8(tensor: torch.Tensor) -> torch.ByteTensor:
    tensor = tensor * 255.0
    tensor.clamp_(0, 255.0)
    return tensor.to(torch.uint8)


def torch_rgb_tensor_to_qimage(tensor: torch.Tensor) -> QImage:
    assert len(tensor.shape) == 3 and tensor.shape[0] == 3, "Bad image format"
    if tensor.is_quantized:
        tensor = tensor.dequantize()
    channels, height, width = tensor.shape
    tensor = __torch_tensor_to_uint8(tensor)
    np_image = __torch_tensor_to_numpy(tensor)
    np_image2 = numpy.require(np_image, numpy.uint8, "C")
    return QImage(np_image2.data, width, height, channels * width, QImage.Format_RGB888)


def torch_mono_tensor_to_qimage(tensor: torch.Tensor) -> QImage:
    assert len(tensor.shape) == 3 and tensor.shape[0] == 1, "Bad image format"
    if tensor.is_quantized:
        tensor = tensor.dequantize()
    tensor3 = torch.cat([tensor, tensor, tensor], dim=0)
    return torch_rgb_tensor_to_qimage(tensor3)
