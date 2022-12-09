"""
Tools to read and write y4m video format from and to pytorch tensors.
For now, it will only support the C420jpeg Ip A0:0 format.

y4m first line is the header:
YUV4MPEG2 W{width} H{height} F{num}:{den} Ip A0:0 C420jpeg

Following the header is any number of frames coded in YCbCr format in Y-Cb-Cr plane order.
Each frame begins with the 5 bytes 'FRAME' followed by zero or more parameters each preceded by
0x20, ending with 0x0A. This is then followed by the raw bytes from each plane.

for more precision about the y4m format: https://wiki.multimedia.cx/index.php/YUV4MPEG2
"""
import re
import torch
import numpy
from typing import BinaryIO, Iterator, Optional, Union, Tuple
from contextlib import contextmanager

YUV4MPEG2_TAG = "YUV4MPEG2"


class Y4MHeader:
    def __init__(self, height: int, width: int, fps_num: int, fps_den: int) -> None:
        self.height = height
        self.width = width
        self.fps_num = fps_num
        self.fps_den = fps_den


def _extract_from_header(tag: str, header: str, data: str = r"(\w+)"):
    return (re.compile(rf"\s{tag}{data}(?:\s|$)").findall(header))[0]


def _check_file(f: BinaryIO) -> bool:
    """Extract data from the header string"""
    current_pos = f.tell()
    f.seek(0)
    header = f.readline().decode()
    f.seek(current_pos)
    var_I = _extract_from_header("I", header)
    var_C = _extract_from_header("C", header)
    assert header.startswith(YUV4MPEG2_TAG), "File is not a YUV4MPEG2 format: " + header
    assert var_I == "p", f"File not Ip: {var_I}"
    assert var_C in ["420jpeg", "420mpeg2"], f"File not C420jpeg: {var_C}"


def _read_header(firstline: bytes) -> Y4MHeader:
    """Extract data from the header string"""
    header = firstline.decode()
    width = _extract_from_header("W", header)
    height = _extract_from_header("H", header)
    (fpsnum, fpsden) = _extract_from_header("F", header, r"(\d+):(\d+)")
    return Y4MHeader(int(height), int(width), int(fpsnum), int(fpsden))


def _decode_frame_420(buffer: bytes, height, width) -> torch.ByteTensor:
    # Convert to Tensor
    frame = numpy.copy(numpy.frombuffer(buffer, dtype=numpy.uint8))
    frame = frame.reshape((height + height // 2, width))
    frame = torch.ByteTensor(frame)
    # Convert to YCbCr420 with (6,h//2,w//2) format
    y = frame[0:height, :].unsqueeze(0)
    cbcr = frame[height:, :].view(2, height // 2, width // 2)
    y = torch.pixel_unshuffle(y, 2)
    return torch.cat([y, cbcr], dim=0)


def _encode_frame_420(frame: torch.ByteTensor) -> bytes:
    assert len(frame.shape) == 3
    # Cat dimensions
    y = frame[0:4, :, :]
    y = torch.pixel_shuffle(y, 2)
    cbcr = frame[4:6, :, :].view(1, frame.shape[1], frame.shape[2] * 2)
    ycbcr = torch.cat([y, cbcr], dim=1)
    # Convert to bytes
    return ycbcr.flatten().cpu().numpy().tobytes()


class Y4MReader:
    def __init__(self, filepath: str) -> None:
        self.__frames_count: Optional[int] = None
        self.__current_frame: int = 0
        # Open and check file
        self.__filepath = filepath
        self.__file = open(self.__filepath, "rb")
        _check_file(self.__file)
        # Read header
        headerline = self.__file.readline()
        self.__afterheader_pos = self.__file.tell()
        self.__header = _read_header(headerline)
        self.__framesize = (self.__header.width // 2) * (self.__header.height // 2) * 6

    def close(self):
        self.__file.close()

    @property
    def header(self) -> Y4MHeader:
        """Header informations"""
        return self.__header

    @property
    def position(self) -> int:
        """Current cursor position"""
        return self.__current_frame

    @property
    def frames_count(self) -> int:
        """Lazy computation of the number of frames"""
        if self.__frames_count is not None:
            return self.__frames_count
        self.__frames_count = self.__compute_frames_count()
        return self.__frames_count

    @property
    def frame_shape(self) -> torch.Size:
        return torch.Size([6, self.__header.height // 2, self.__header.width // 2])

    @property
    def colorspace(self) -> str:
        return "ycbcr420"

    def seek(self, position: int):
        assert position < self.frames_count, "Cannot seek further than the last frame."
        if position < 0:
            position = position % self.frames_count
        if position == 0 or position < self.__current_frame:
            self.__file.seek(self.__afterheader_pos)
            self.__current_frame = 0
        while self.__current_frame < position:
            self.skip_frame()

    def read_frame(self) -> torch.ByteTensor:
        frame_header = self.__file.readline().decode()
        if not frame_header.startswith("FRAME"):
            raise EOFError()
        buffer = self.__file.read(self.__framesize)
        self.__current_frame += 1
        return _decode_frame_420(buffer, self.__header.height, self.__header.width)

    def skip_frame(self) -> torch.ByteTensor:
        frame_header = self.__file.readline()
        if not frame_header.decode().startswith("FRAME"):
            raise EOFError()
        self.__file.read(self.__framesize)
        self.__current_frame += 1

    def __len__(self) -> torch.ByteTensor:
        return self.frames_count

    def __getitem__(self, idx: Union[int, slice]) -> torch.ByteTensor:
        # One frame
        if isinstance(idx, int):
            self.seek(idx)
            return self.read_frame()
        # Handle slicing
        assert isinstance(idx, slice)
        start = idx.start % self.frames_count if idx.start is not None else 0
        stop = idx.stop % self.frames_count if idx.stop is not None else self.frames_count
        step = idx.step if idx.step is not None else 1
        assert start <= stop, "Looped slice are not authorized yet"
        # Concat concerned frames
        frames = []
        for i in range(start, stop, step):
            self.seek(i)
            frames.append(self.read_frame().unsqueeze(0))
        return torch.cat(frames, dim=0)

    def iter(self) -> Iterator[torch.ByteTensor]:
        while True:
            try:
                yield self.read_frame()
            except EOFError:
                break

    def __compute_frames_count(self) -> int:
        # Move to the start frame
        current_pos = self.__file.tell()
        self.__file.seek(self.__afterheader_pos)
        # Count every frames
        frames_count: int = 0
        while True:
            try:
                self.skip_frame()
            except EOFError:
                break
            frames_count += 1
        # Go back to previous position
        self.__file.seek(current_pos)
        return frames_count


class Y4MWriter:
    def __init__(
        self,
        filepath: str,
        w: int,
        h: int,
        f: Tuple[int, int],
        i: str = "p",
        a: Tuple[int, int] = (0, 0),
        override: bool = False,
    ):
        assert w % 2 == 0
        assert h % 2 == 0
        self.__shape: torch.Size = torch.Size([6, w // 2, h // 2])
        self.__cursor: int = 0
        self.__framesize = (w // 2) * (h // 2) * 6
        # Open file
        self.__filepath = filepath
        self.__file = open(self.__filepath, "wb+" if override else "wb")
        # Write header
        header = f"YUV4MPEG2 W{w} H{h} F{f[0]}:{f[1]} I{i} A{a[0]}:{a[1]} C420jpeg Xkompil\n"
        self.__file.write(header.encode())

    def close(self):
        self.__file.close()

    def write_frame(self, frame: torch.ByteTensor):
        self.__file.write(b"FRAME\n")
        self.__file.write(_encode_frame_420(frame))


@contextmanager
def read_y4m(filepath: str) -> Iterator[Y4MReader]:
    f = Y4MReader(filepath)
    try:
        yield f
    finally:
        f.close()


@contextmanager
def write_y4m(
    filepath: str,
    w: int,
    h: int,
    f: Tuple[int, int],
    i: str = "p",
    a: Tuple[int, int] = (0, 0),
    override: bool = False,
) -> Iterator[Y4MWriter]:
    f = Y4MWriter(filepath, w, h, f, i, a, override)
    try:
        yield f
    finally:
        f.close()
