import torch

try:
    from kompil_utils import graycode, feed_binary_tensor, layer_graycode, layer_binary

    has_kompil_ext = True
except ImportError as exception:
    print("WARNING: kompil utils extension not imported. Python implementation will replace cpp.")
    has_kompil_ext = False


class _CppImplem:
    @staticmethod
    def graycode(n: int) -> int:
        return graycode(n)

    @staticmethod
    def feed_binary_tensor(bin_value: int, t_out: torch.Tensor):
        return feed_binary_tensor(bin_value, t_out)

    @staticmethod
    def layer_graycode(t_val: torch.Tensor, t_out: torch.Tensor):
        layer_graycode(t_val, t_out)

    @staticmethod
    def layer_binary(t_val: torch.Tensor, t_out: torch.Tensor):
        layer_binary(t_val, t_out)


class _PythonImplem:
    @staticmethod
    def graycode(n: int) -> int:
        return n ^ (n >> 1)

    @staticmethod
    def feed_binary_tensor(bin_value: int, t_out: torch.Tensor):
        t_out.fill_(0)
        bitsize = t_out.shape[0]
        mask = 1
        bit = bitsize - 1
        while True:
            bitval = bin_value & mask
            if bool(bitval):
                t_out[bit] = 1
            if bit == 0:
                break
            bit -= 1
            mask = mask << 1

    @staticmethod
    def layer_graycode(t_val: torch.Tensor, t_out: torch.Tensor):
        for b in range(t_out.shape[0]):
            bin_value = t_val[b][0].long().item()
            gray_code = _PythonImplem.graycode(bin_value)
            t_out_spec = t_out[b]
            _PythonImplem.feed_binary_tensor(gray_code, t_out_spec)

    @staticmethod
    def layer_binary(t_val: torch.Tensor, t_out: torch.Tensor):
        for b in range(t_out.shape[0]):
            bin_value = t_val[b][0].long().item()
            t_out_spec = t_out[b]
            _PythonImplem.feed_binary_tensor(bin_value, t_out_spec)


if not has_kompil_ext:
    graycode = _PythonImplem.graycode
    feed_binary_tensor = _PythonImplem.feed_binary_tensor
    layer_graycode = _PythonImplem.layer_graycode
    layer_binary = _PythonImplem.layer_binary


def get_gc_nodes(nb_frames: int) -> int:
    """
    Return the number of expected nodes in the timeline.
    """
    return len(bin(graycode(nb_frames - 1))[2:])
