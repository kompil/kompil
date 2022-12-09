from kompil.utils.y4m import Y4MReader
from kompil.player.decoders.decoder import register_decoder, Decoder
from kompil.player.options import PlayerTargetOptions


@register_decoder("y4m")
class Y4MDecoder(Decoder):
    @classmethod
    def generate(cls, file: str, opt: PlayerTargetOptions) -> Decoder:
        return cls(file, opt)

    def __init__(self, fpath, opt: PlayerTargetOptions):
        self.__opt = opt
        self.__reader = Y4MReader(fpath)
        self.__fps = self.__reader.header.fps_num / self.__reader.header.fps_den
        self._pos = 0

    def __del__(self):
        self.__reader.close()

    def get_total_frames(self):
        return self.__reader.frames_count

    def get_framerate(self):
        return self.__fps

    def get_cur_frame(self):
        return self.__reader[self._pos].to(self.__opt.device).to(self.__opt.dtype) / 255.0

    def set_position(self, pos):
        self._pos = pos

    def get_position(self) -> int:
        return self._pos

    def get_colorspace(self) -> str:
        return self.__reader.colorspace
