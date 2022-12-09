import torch

from kompil.data.timeline import create_timeline
from kompil.player.decoders.decoder import register_decoder, Decoder
from kompil.player.options import PlayerTargetOptions
from kompil.packers.utils import generic_load


@register_decoder("kompil")
class KompilDecoder(Decoder):
    @classmethod
    def generate(cls, fpath: str, opt: PlayerTargetOptions) -> Decoder:
        return cls(fpath, opt.device, opt.dtype)

    def __init__(self, filename, device: torch.device, dtype: torch.dtype):
        self._frame_pos = 0
        self._dtype = dtype
        # Select device
        self._device = device
        if not torch.cuda.is_available():
            if self._device == torch.device("cuda"):
                print("WARNING: cuda not available, switching to cpu")
            self._device = torch.device("cpu")
            self._dtype = torch.float32
        # Load model
        self._model = generic_load(filename).to(self._dtype).to(self._device)
        self._fps = self._model.fps
        self._nb_frames = self._model.nb_frames
        # Timeline
        self.__timeline = create_timeline(self._nb_frames, device=self._device).to(self._dtype)
        # Buffer
        self.__last_frame = None

    def get_total_frames(self):
        return self._nb_frames

    def get_framerate(self):
        return self._fps

    def __calculate_frame(self, idx: int) -> torch.Tensor:
        with torch.no_grad():
            time_vec = self.__timeline[idx]
            return self._model.forward(time_vec.unsqueeze(0))[0]

    def get_cur_frame(self):
        if self.__last_frame is None or self.__last_frame[0] != self._frame_pos:
            frame = self.__calculate_frame(self._frame_pos)
            self.__last_frame = (self._frame_pos, frame)
            return frame
        return self.__last_frame[1]

    def set_position(self, pos):
        self._frame_pos = pos

    def get_position(self) -> int:
        return self._frame_pos

    def get_colorspace(self) -> str:
        return self._model.colorspace
