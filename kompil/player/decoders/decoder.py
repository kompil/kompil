import abc
import torch

from kompil.utils.factory import Factory
from kompil.player.options import PlayerTargetOptions

__FACTORY = Factory("player-decoders")


def decoder_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_decoder(name: str):
    return decoder_factory().register(name)


class Decoder(abc.ABC):
    @abc.abstractmethod
    def get_total_frames(self) -> int:
        pass

    @abc.abstractmethod
    def get_framerate(self):
        pass

    @abc.abstractmethod
    def get_cur_frame(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def set_position(self, pos):
        pass

    @abc.abstractmethod
    def get_position(self) -> int:
        pass

    @abc.abstractmethod
    def get_colorspace(self) -> str:
        pass


def __auto_decoder(file_path: str):
    """Select decoder based on the file extension"""
    if file_path.endswith(".pth"):
        return "kompil"
    if file_path.endswith(".y4m"):
        return "y4m"
    return "decord"


def create_decoder(decoder: str, fpath: str, opt: PlayerTargetOptions):
    """Create a decoder"""
    if decoder == "auto":
        decoder = __auto_decoder(fpath)

    dec_cls = decoder_factory().get(decoder)
    return dec_cls.generate(fpath, opt)
