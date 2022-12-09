import subprocess
from enum import Enum


class CodecLibs(Enum):
    H264 = "h264"
    X264 = "libx264"
    H265 = "libx265"
    XVID = "libxvid"
    MPEG4 = "mpeg4"
    CODEC_MJPEG = "mjpeg"
    CODEC_VP9 = "libvpx-vp9"
    CODEC_AV1 = "libaom-av1"


def __run_ffmpeg_cmd(args: str) -> str:
    assert args

    # return subprocess.check_output(f"ffmpeg {args}", shell=True).decode("utf-8")

    return subprocess.check_output(f"ffmpeg {args}", shell=True, stderr=subprocess.STDOUT).decode(
        "utf-8"
    )


def transcode_to_mpeg4(video_fpath: str, target_fpath: str, quality: int):
    """Quality from 1 to 31"""
    __run_ffmpeg_cmd(f"-i {video_fpath} -c:v mpeg4 -q:v {quality} {target_fpath}")


def transcode_to_xvid(video_fpath: str, target_fpath: str, quality: int):
    """Quality from 1 to 31"""
    __run_ffmpeg_cmd(f"-i {video_fpath} -c:v libxvid -q:v {quality} {target_fpath}")


def transcode_to_mpeg2(video_fpath: str, target_fpath: str, quality: int):
    """Quality from 0 to 51, default would be 23"""
    __run_ffmpeg_cmd(f"-i {video_fpath} -c:v mpeg2video -qscale:v {quality} {target_fpath} -y")


def transcode_to_h264(video_fpath: str, target_fpath: str, quality: int):
    """Quality from 0 to 51, default would be 23"""
    preset = "-preset veryslow"
    __run_ffmpeg_cmd(f"-i {video_fpath} {preset} -c:v libx264 -crf {quality} {target_fpath}")


def transcode_to_vp9(video_fpath: str, target_fpath: str, quality: int):
    """Quality from 0 to 51, default would be 23"""
    __run_ffmpeg_cmd(f"-i {video_fpath} -c:v libvpx-vp9 -crf {quality} {target_fpath}")


def transcode_to_y4m(video_fpath: str, target_fpath: str):
    assert target_fpath.endswith(".y4m")
    __run_ffmpeg_cmd(f"-i {video_fpath} {target_fpath}")
