import abc
import torch
import types

from kompil.utils.factory import Factory
from kompil.data.video import VideoLoader


MASKTYPE = torch.float16

__FACTORY = Factory("maskmaker")


def maskmaker_factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_maskmaker(name: str, *args, **kwargs):
    def to_class(fct):
        if isinstance(fct, types.FunctionType):
            assert not args, name + " function maskmaker can't have *args"
            return MaskMakerFunction, [fct], kwargs
        return fct, args, kwargs

    return maskmaker_factory().register(name, hook=to_class)


def instantiate_maskmaker(name, **kwargs):
    cls, args, base_kwargs = maskmaker_factory().get(name)
    base_kwargs.update(kwargs)
    return cls(*args, **base_kwargs)


class MaskMakerBase(abc.ABC):
    @abc.abstractmethod
    def generate(self, video_loader: VideoLoader) -> torch.HalfTensor:
        pass


class AutoMaskMakerBase(MaskMakerBase):
    def generate(self, video_loader: VideoLoader) -> torch.HalfTensor:

        print("Initializing the maskmaker...")
        self.init(nb_frames=len(video_loader), frame_shape=video_loader.frame_shape)

        for idx, frame in enumerate(video_loader):
            completion = idx / (len(video_loader) - 1)
            print(f"Reading frames... {completion * 100.0:0.1f}%  ", end="\r")
            self.push_frame(frame)

        print()

        print("Computing the mask...")
        mask = self.compute()

        print("Done")
        return mask

    @abc.abstractmethod
    def init(self, nb_frames: int, frame_shape: torch.Size):
        pass

    @abc.abstractmethod
    def push_frame(self, frame: torch.Tensor):
        pass

    @abc.abstractmethod
    def compute(self) -> torch.HalfTensor:
        pass


class MaskMakerFunction(AutoMaskMakerBase):
    def __init__(self, fct, **kwargs):
        self.__mask = None
        self.__fct = fct
        self.__kwargs = kwargs

    def init(self, nb_frames: int, frame_shape: torch.Size):
        self.__mask = []

    def push_frame(self, frame: torch.Tensor):
        # Do nothing
        self.__mask.append(self.__fct(frame, **self.__kwargs).unsqueeze(0))

    def compute(self) -> torch.HalfTensor:
        return torch.cat(self.__mask, dim=0)
