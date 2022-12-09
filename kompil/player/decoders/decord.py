import decord

from kompil.utils.video import decord_to_tensor
from kompil.player.decoders.decoder import register_decoder, Decoder
from kompil.player.options import PlayerTargetOptions


@register_decoder("decord")
class DecordDecoder(Decoder):
    @classmethod
    def generate(cls, fpath: str, opt: PlayerTargetOptions) -> Decoder:
        return cls(fpath)

    def __init__(self, filename):
        self._capture = decord.VideoReader(uri=filename, ctx=decord.cpu())
        self._pos = 0

    def get_total_frames(self):
        return len(self._capture)

    def get_framerate(self):
        return self._capture.get_avg_fps()

    def get_cur_frame(self):
        return decord_to_tensor(self._capture[self._pos])

    def set_position(self, pos):
        self._pos = pos

    def get_position(self) -> int:
        return self._pos

    def get_colorspace(self) -> str:
        return "rgb8"
