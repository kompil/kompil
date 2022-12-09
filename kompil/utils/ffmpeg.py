import os
import subprocess
from typing import Tuple

CODEC_H264 = "h264"
CODEC_H265 = "libx265"
CODEC_MP4 = "mpeg4"
CODEC_MJPEG = "mjpeg"
CODEC_VP9 = "libvpx-vp9"
CODEC_AV1 = "libaom-av1"


def __run_ffprobe_cmd(args: str) -> str:
    assert args

    return subprocess.check_output(f"ffprobe {args}", shell=True, stderr=subprocess.STDOUT).decode(
        "utf-8"
    )


def __run_ffmpeg_cmd(args: str) -> str:
    assert args

    return subprocess.check_output(f"ffmpeg {args}", shell=True, stderr=subprocess.STDOUT).decode(
        "utf-8"
    )


def get_duration(video_path: str) -> float:
    assert os.path.exists(video_path)

    result = __run_ffprobe_cmd(
        f"-v error -select_streams v:0 -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_path}"
    )

    return float(result)


def get_resolution(video_path: str) -> Tuple[int, int]:
    assert os.path.exists(video_path)

    result = __run_ffprobe_cmd(
        f"-v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {video_path}"
    )

    if not result:
        return 0, 0

    wh = result.split(",")

    return int(wh[1]), int(wh[0])  # height, width


def get_bitrate(video_path: str) -> int:
    assert os.path.exists(video_path)

    result = __run_ffprobe_cmd(
        f"-v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 {video_path}"
    )

    try:
        return int(result)
    except:
        return 0


def get_framerate(video_path: str) -> int:
    assert os.path.exists(video_path)

    result = __run_ffprobe_cmd(
        f"-v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {video_path}"
    )

    vals = result.split("/")

    return round(int(vals[0]) / int(vals[1]))


def resize_video(original_video_path: str, dest_video_path: str, width: int, height: int) -> str:
    assert original_video_path
    assert dest_video_path
    assert width
    assert height

    __run_ffmpeg_cmd(
        f'-i {original_video_path} -filter:v scale="{width}:{height}" -c:a copy {dest_video_path}'
    )

    return dest_video_path


def convert_video_to_mjpeg(original_video_path: str, dest_video_path: str) -> str:
    assert original_video_path
    assert dest_video_path

    __run_ffmpeg_cmd(
        f"-i {original_video_path} -vcodec {CODEC_MJPEG} -qscale 1 -an {dest_video_path}"
    )

    return dest_video_path


def convert_video_to_vp9(original_video_path: str, dest_video_path: str) -> str:
    assert original_video_path
    assert dest_video_path

    __run_ffmpeg_cmd(f"-i {original_video_path} -c:v {CODEC_VP9} -crf 30 -b:v 0 {dest_video_path}")

    return dest_video_path


def convert_video_to_av1(original_video_path: str, dest_video_path: str) -> str:
    assert original_video_path
    assert dest_video_path

    __run_ffmpeg_cmd(
        f"-i {original_video_path} -c:v {CODEC_AV1} -strict experimental -crf 30 -b:v 0 -row-mt 1 -threads 8 -cpu-used 4 {dest_video_path}"
    )

    return dest_video_path


def change_video_fps(original_video_path: str, dest_video_path: str, fps: int = 30) -> str:
    assert original_video_path
    assert dest_video_path
    assert fps > 0

    __run_ffmpeg_cmd(f"-i {original_video_path} -filter:v fps={fps} {dest_video_path}")

    return dest_video_path


def cut_video(
    original_video_path: str, dest_video_path: str, start_time_sec: int, length_sec: int
) -> str:
    assert original_video_path
    assert dest_video_path
    assert length_sec
    assert get_duration(original_video_path) > start_time_sec + length_sec

    __run_ffmpeg_cmd(
        f"-ss {start_time_sec} -i {original_video_path} -c copy -t {length_sec} {dest_video_path}"
    )

    return dest_video_path


def create_video_from_frames(
    video_filepath: str,
    frame_folder: str,
    frame_prefix: str = "frame_",
    codec: str = CODEC_H265,
    fps: int = 30,
    crf: int = 30,
) -> str:
    """
    Create a video with ffmpeg CLI tool from a image folder that must be in the format : [frame_prefix]%d.png

    :return: Fullpath of the generated video
    """

    assert frame_folder
    assert codec

    os.system(f"cd {frame_folder}")
    __run_ffmpeg_cmd(
        f"-framerate {fps} -start_number 0 -i {frame_folder}/{frame_prefix}%d.png "
        f"-c:v {codec} -crf {crf} -b:v 0 -qscale:v 1 {video_filepath}"
    )

    return video_filepath


def compute_psnr_ssim(original_video_path: str, other_video_path: str) -> Tuple[str, str]:
    """
    Compute PSNR and SSIM between two video files

    :return: Tuple (psnr, ssim) of string (containing all info)
    """

    assert original_video_path
    assert other_video_path
    assert os.path.isfile(original_video_path)
    assert os.path.isfile(other_video_path)

    output = __run_ffmpeg_cmd(
        f'-i {original_video_path} -i {other_video_path} -lavfi "ssim;[0:v][1:v]psnr" -f null - '
    )

    assert output

    if "\n" in output:
        output = output.replace("\r", "")
    else:
        output = output.replace("\r", "\n")

    splitted_output = str(output).split("\n")
    psnr_line = splitted_output[-2]
    ssim_line = splitted_output[-3]

    return psnr_line, ssim_line


def extract_audio(original_video_path: str, dest_audio_path: str) -> str:
    assert original_video_path
    assert dest_audio_path

    __run_ffmpeg_cmd(f"-i {original_video_path} -q:a 0 -map a {dest_audio_path}")

    return dest_audio_path


def cut_audio(
    original_audio_path: str, dest_audio_path: str, start_time_sec: int, length_sec: int
) -> str:
    assert original_audio_path
    assert dest_audio_path
    assert length_sec

    __run_ffmpeg_cmd(
        f"-ss {start_time_sec} -i {original_audio_path} -c copy -t {length_sec} {dest_audio_path}"
    )

    return dest_audio_path
