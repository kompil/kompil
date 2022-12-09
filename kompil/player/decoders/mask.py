import torch

from kompil.player.decoders.decoder import register_decoder, Decoder
from kompil.player.options import PlayerTargetOptions


@register_decoder("mask")
class MaskDecoder(Decoder):
    @classmethod
    def generate(cls, file: str, opt: PlayerTargetOptions) -> Decoder:
        return cls(file, opt)

    def __init__(self, fpath, opt: PlayerTargetOptions):
        self.__mask = torch.load(fpath).to(opt.device).to(opt.dtype)
        self.__fps = None
        self._pos = 0

    def get_total_frames(self):
        return self.__mask.shape[0]

    def get_framerate(self):
        return self.__fps

    def get_cur_frame(self):
        return self.__mask[self._pos]

    def set_position(self, pos):
        self._pos = pos

    def get_position(self) -> int:
        return self._pos

    def get_colorspace(self) -> str:
        return "yuv420"
