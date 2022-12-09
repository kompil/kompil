import os
import cv2
import decord
import torch
from typing import Optional, Tuple, Union

from kompil.utils.ffmpeg import get_duration, get_resolution, get_bitrate, get_framerate


RESOLUTION_MAP = {
    "100p": (3, 100, 178),
    "178p": (3, 178, 320),
    "320p": (3, 320, 480),
    "480p": (3, 480, 720),
    "720p": (3, 720, 1280),
    "1080p": (3, 1080, 1920),
    "1440p": (3, 1440, 2560),
    "2160p": (3, 2160, 3840),
    "4320p": (3, 4320, 7680),
    "1920x880": (3, 880, 1920),
}

RESOLUTION_MAP_INVERSE = {(value[1], value[2]): key for key, value in RESOLUTION_MAP.items()}


def discretize(img: torch.Tensor):
    img_255 = img * 255.0  # up to [0; 255] range like byte-storage version

    floor_255 = torch.floor(img_255)  # considere floor because of float to byte conversion

    disc_img = floor_255 / 255.0  # back to [0; 1] range

    return disc_img


def chw_to_resolution(chw: tuple) -> str:
    assert chw

    length = len(chw)

    assert length == 2 or length == 3

    if length == 3:
        chw = chw[1:]

    return RESOLUTION_MAP_INVERSE[chw]


def resolution_to_chw(resolution: str) -> Tuple[int, int, int]:
    assert resolution
    if resolution == "-1":
        return (-1, -1, -1)
    return RESOLUTION_MAP[resolution]


def get_video_frame_info(video_path: str, resolution: Union[str, None] = None):
    reader = decord.VideoReader(uri=video_path, ctx=decord.cpu())
    first_frame: torch.Tensor = reader[0]
    h, w, c = first_frame.shape
    frames = len(reader)

    if resolution:
        c, h, w = resolution_to_chw(resolution)

    return frames, c, h, w


def get_video_data(video_path: str):
    assert os.path.exists(video_path)

    duration = int(get_duration(video_path))
    _, file_extension = os.path.splitext(video_path)
    resolution = chw_to_resolution(get_resolution(video_path))
    fps = get_framerate(video_path)
    bitrate = get_bitrate(video_path)
    file_size = os.path.getsize(video_path)

    return duration, fps, file_extension, resolution, bitrate, file_size


def decord_to_tensor_permute(frame: torch.Tensor):
    # bhwc to bchw
    if len(frame.shape) == 4:
        frame = frame.permute(0, 3, 1, 2)
    # hwc to chw
    elif len(frame.shape) == 3:
        frame = frame.permute(2, 0, 1)
    return frame


def decord_to_tensor_float(frame: torch.Tensor, half: bool = False):
    # Convert it to [0.0, 1.0] float
    original_type = frame.dtype
    if half:
        frame = frame.half()
    else:
        frame = frame.float()
    frame = frame / 255.0

    # clamp output, no need if it came from uint8
    if original_type != torch.uint8:
        frame = torch.clamp(frame, min=0.0, max=1.0)

    return frame


def decord_to_numpy(frame):
    np_frame = frame.cpu().type(torch.uint8).numpy()
    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)

    return np_frame


def decord_to_tensor(frame, device=None, half: bool = False):
    # https://github.com/dmlc/decord
    # frame = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_BGR2RGB) #if enabled, remove asnumpy() in next line and save_frame
    if not isinstance(frame, torch.Tensor):
        frame = torch.Tensor(frame.asnumpy())

    # Move it to the target device
    if device:
        frame = frame.to(device)

    # Apply modifications
    frame = decord_to_tensor_permute(frame)
    frame = decord_to_tensor_float(frame, half)

    return frame


def numpy_to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # needed if using decord (based on BGR)
    frame = torch.Tensor(frame).float()

    # hwc to chw
    frame = frame.permute(2, 0, 1)

    return torch.clamp(frame / 255.0, min=0.0, max=1.0)


def tensor_to_numpy(frame: torch.Tensor):
    frame = frame * 255.0

    # If black image
    if len(frame.shape) == 2:
        return frame.cpu().type(torch.uint8).numpy()

    # chw to hwc
    frame = frame.permute(1, 2, 0)

    np_frame = frame.cpu().type(torch.uint8).numpy()
    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)  # needed if using decord (based on BGR)

    return np_frame


def display_frame(frame: torch.Tensor, window: str = "display", wait_time: int = 0):
    np_frame = tensor_to_numpy(frame)
    cv2.imshow(window, np_frame)

    return cv2.waitKey(wait_time) & 0xFF == ord("q")


def display_video(video: torch.Tensor, name: str = "video", loop: bool = False):
    first = True

    while first or loop:
        first = False
        for i in range(video.shape[0]):
            print(f"Playing video {i+1}/{video.shape[0]}", end="\r")
            if display_frame(video[i], name, 33):
                print()
                return
        print()

        if loop:
            print("play again...")


def create_frames(video, folder: str, frame_prefix: str = "frame_", start_id: int = 0):
    os.makedirs(folder, exist_ok=True)

    for i, frame in enumerate(video):
        actual_frame_id = start_id + i
        print("Generating frame: ", actual_frame_id + 1, end="\r")
        filepath = os.path.join(folder, f"{frame_prefix}{actual_frame_id}.png")
        save_frame(frame, filepath)

    print()

    return folder


def save_frame(frame: torch.Tensor, filepath: str):
    frame.clamp_(0.0, 1.0)
    np_frame = tensor_to_numpy(frame)
    cv2.imwrite(filepath, np_frame)


def load_frame(filepath: str, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    np_frame = cv2.imread(filepath)

    if np_frame is None:
        raise FileNotFoundError(f"File {filepath} not found.")

    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)  # needed if using decord (based on BGR)
    frame = torch.Tensor(np_frame)

    frame = frame.permute(2, 0, 1)

    if out is None:
        return frame

    out.copy_(frame)
    return frame
