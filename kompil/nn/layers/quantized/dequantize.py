import torch


class DeQuantize(torch.nn.Module):
    def __init__(self):
        super(DeQuantize, self).__init__()

    def forward(self, x):
        return x.dequantize()

    @staticmethod
    def from_float(mod):
        return DeQuantize()
