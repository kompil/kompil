import os
import abc
import torch
import decord
import threading
from typing import Union, Tuple, List, Optional

from kompil.utils.video import resolution_to_chw
from kompil.utils.y4m import read_y4m, Y4MReader
from kompil.utils.colorspace import convert_to_colorspace, convert_shape_to_colorspace


class VideoLoader(abc.ABC):
    @property
    @abc.abstractmethod
    def file_path(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def colorspace(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def fps(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def frame_shape(self) -> torch.Size:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_raw(self, idx: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def convert_raw(self, frame: torch.Tensor) -> torch.Tensor:
        pass


def _resize_yuv420(frames: torch.ByteTensor, resolution) -> torch.ByteTensor:
    if resolution is None or resolution == "-1":
        return frames
    _, req_h, req_w = resolution_to_chw(resolution)
    _, _, h, w = frames.shape
    if req_h == h and req_w == w:
        return frames
    mode = "bilinear"
    frames = frames.float()
    y = frames[:, 0:4, :, :]
    y = torch.pixel_shuffle(y, 2)
    y = torch.nn.functional.interpolate(y, size=(req_h, req_w), mode=mode)
    y = torch.pixel_unshuffle(y, 2)
    uv = frames[:, 4:6, :, :]
    uv = torch.nn.functional.interpolate(uv, size=(req_h // 2, req_w // 2), mode=mode)
    yuv = torch.cat([y, uv], dim=1)
    return yuv.byte()


class MockVideoLoader(VideoLoader):
    """
    Video loader that loads the data of every frames in the storage device. Then send it to the
    target device and convert it to the right type just in time.

    :note: It does only handle y4m format right now.
    """

    def __init__(
        self,
        resolution: str,
        colorspace: str,
        target_device: torch.device,
    ):
        self.__fps = 24.0
        self.__res = convert_shape_to_colorspace(resolution_to_chw(resolution), "rgb8", colorspace)
        self.__len = torch.tensor([int(self.__fps * 2)])  # 2 seconds
        self.__target_device = target_device
        self.__image_tensor = torch.randint(
            0,
            255,
            self.__res,
            dtype=torch.uint8,
            device=target_device,
        )
        self.__colorspace = colorspace
        self.__frame_shape = self.convert_raw(self.__image_tensor).shape

    @property
    def file_path(self) -> str:
        return "mock.mock"

    @property
    def colorspace(self) -> str:
        return self.__colorspace

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def frame_shape(self) -> torch.Size:
        return self.__frame_shape

    @property
    def device(self) -> torch.device:
        return self.__target_device

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        return self.convert_raw(self.get_raw(idx))

    def get_raw(self, idx: Union[int, slice]) -> torch.Tensor:
        if isinstance(idx, int):
            return self.__image_tensor
        actual_len = len(range(*idx.indices(self.__len)))
        return self.__image_tensor.repeat(actual_len, *self.__frame_shape)

    def convert_raw(self, frame: torch.Tensor) -> torch.Tensor:
        # Move to float type
        ycbcr420 = frame.to(torch.float)
        # Rescale
        ycbcr420 = ycbcr420 / 255.0
        # Already in targeted colorspace
        return ycbcr420


class FullVideoLoader(VideoLoader):
    """
    Video loader that loads the data of every frames in the storage device. Then send it to the
    target device and convert it to the right type just in time.

    :note: It does only handle y4m format right now.
    """

    def __init__(
        self,
        video_file_path: str,
        colorspace: str,
        storage_device: torch.device = None,
        target_device: torch.device = None,
        pin_memory: bool = False,
        resolution: Union[str, None] = None,
        index_table: Optional[List[int]] = None,
    ):
        self.__file_path = video_file_path
        self.__video, self.__fps, self.__src_colorspace = self.__load(
            video_file_path, storage_device, pin_memory, resolution, index_table
        )
        self.__len = self.__video.shape[0]
        self.__target_device = target_device if target_device else self.__video.device
        self.__colorspace = colorspace
        self.__frame_shape = self.convert_raw(self.__video[0]).shape

    def __load(
        self,
        file_path: str,
        device: torch.device,
        pin_memory: bool,
        resolution: Union[str, None],
        index_table: Optional[List[int]],
    ) -> Tuple[torch.Tensor, int]:
        """
        :return: full video, fps, colorspace
        """

        with read_y4m(filepath=file_path) as y4m:
            # Get video info
            fc = y4m.frames_count
            colorspace = y4m.colorspace
            index_table = index_table if index_table is not None else list(range(fc))
            # Verify compat
            assert colorspace == "ycbcr420", "Only handle ycbcr420 colorspace"
            # Read the frames
            with torch.no_grad():
                frames = []
                for frame_id in index_table:
                    frame = y4m[frame_id]
                    frame = _resize_yuv420(frame.unsqueeze(0), resolution)
                    frame = frame.to(device)
                    frames.append(frame)
                video = torch.cat(frames, dim=0)
            # Get FPS
            fps = y4m.header.fps_num / y4m.header.fps_den

        # Move to pin memory if asked
        video = video.pin_memory() if pin_memory else video

        # Return the result
        return video, fps, colorspace

    @property
    def file_path(self) -> str:
        return self.__file_path

    @property
    def colorspace(self) -> str:
        return self.__colorspace

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def frame_shape(self) -> torch.Size:
        return self.__frame_shape

    @property
    def device(self) -> torch.device:
        return self.__target_device

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        return self.convert_raw(self.get_raw(idx))

    def get_raw(self, idx: Union[int, slice]) -> torch.Tensor:
        return self.__video[idx].to(self.__target_device)

    def convert_raw(self, frame: torch.Tensor) -> torch.Tensor:
        # Move to float type
        ycbcr420 = frame.to(torch.float)
        # Rescale
        ycbcr420 = ycbcr420 / 255.0
        # Convert to targeted colorspace
        return convert_to_colorspace(
            ycbcr420, src=self.__src_colorspace, dst=self.__colorspace, lossy_allowed=True
        )


class StreamVideoLoader(VideoLoader):
    """
    Video loader that keep the video file on disk and read it frame per frame.

    :note: It does only handle y4m format right now.
    """

    def __init__(
        self,
        video_file_path: str,
        colorspace: str,
        device: torch.device,
        resolution: str,
        index_table: Optional[List[int]] = None,
        half: bool = False,
    ):
        self.__half = half
        self.__file_path = video_file_path
        self.__resolution = resolution
        self.__device = device
        self.__colorspace = colorspace
        # Start reading the file
        self.__reader = Y4MReader(video_file_path)
        fc = self.__reader.frames_count
        self.__index_table = index_table if index_table is not None else list(range(fc))
        # Verify compat
        assert self.__reader.colorspace == "ycbcr420", "Only handle ycbcr420 colorspace"
        # Get the fps
        self.__fps = self.__reader.header.fps_num / self.__reader.header.fps_den
        # Read data from first frame
        first_frame: torch.Tensor = self.__reader[0]
        if not device:
            self.__device = first_frame.device
        self.__frame_shape = _resize_yuv420(first_frame.unsqueeze(0), self.__resolution).shape[1:]
        # lock for ensuring one frame is read at once
        self.__lock = threading.Lock()

    def __del__(self):
        self.__reader.close()

    @property
    def file_path(self) -> str:
        return self.__file_path

    @property
    def colorspace(self) -> str:
        return self.__colorspace

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def frame_shape(self) -> torch.Size:
        return self.__frame_shape

    @property
    def device(self) -> torch.device:
        return self.__device

    def __len__(self) -> int:
        return len(self.__index_table)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        return self.convert_raw(self.get_raw(idx))

    def get_raw(self, idx: Union[int, slice]) -> torch.Tensor:
        with self.__lock:
            if not isinstance(idx, slice):
                frames = self.__reader[self.__index_table[idx]].unsqueeze(0)
                return _resize_yuv420(frames, self.__resolution)[0]

            actual = self.__index_table[idx]
            tensors = []
            for actual_idx in actual:
                frame = self.__reader[actual_idx]
                frame = _resize_yuv420(frame.unsqueeze(0), self.__resolution)
                frame = frame.to(self.__device)
                tensors.append(frame)
            return torch.cat(tensors, dim=0)

    def convert_raw(self, frame: torch.Tensor) -> torch.Tensor:
        # Move to float type
        if self.__half:
            ycbcr420 = frame.to(torch.half)
        else:
            ycbcr420 = frame.to(torch.float)
        # Rescale
        ycbcr420 = ycbcr420 / 255.0
        # Convert to targeted colorspace
        return convert_to_colorspace(
            ycbcr420, src=self.__reader.colorspace, dst=self.__colorspace, lossy_allowed=True
        )
