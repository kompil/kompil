import os
import torch
from typing import List

import kompil.utils.ffmpeg2 as ffmpeg2
from kompil.profile.bench import bench, Y4mBenchInput
from kompil.utils.resources import get_video
from kompil.eval import write_output_file


TMP_STANDARD_PATH = os.path.join("/dev", "shm", "kompil_std")


def __encode_avc(video_path: str, video_name: str, quality: float) -> str:
    os.makedirs(TMP_STANDARD_PATH, exist_ok=True)
    # Adjust quality
    quality = int((100 - quality) / 100.0 * 51)
    print("Quality converted to", quality)
    # Encode
    encoded_path = os.path.join(TMP_STANDARD_PATH, f"{video_name}_q{quality}.mp4")
    ffmpeg2.transcode_to_h264(video_path, encoded_path, quality=quality)
    # Back to y4m
    lossless_path = os.path.join(TMP_STANDARD_PATH, f"{video_name}_q{quality}.y4m")
    ffmpeg2.transcode_to_y4m(encoded_path, lossless_path)
    return encoded_path, lossless_path


def __encode_vp9(video_path: str, video_name: str, quality: float) -> str:
    os.makedirs(TMP_STANDARD_PATH, exist_ok=True)
    # Adjust quality
    quality = int((100 - quality) / 100.0 * 51)
    print("Quality converted to", quality)
    # Encode
    encoded_path = os.path.join(TMP_STANDARD_PATH, f"{video_name}_q{quality}.webm")
    ffmpeg2.transcode_to_vp9(video_path, encoded_path, quality=quality)
    # Back to y4m
    lossless_path = os.path.join(TMP_STANDARD_PATH, f"{video_name}_q{quality}.y4m")
    ffmpeg2.transcode_to_y4m(encoded_path, lossless_path)
    return encoded_path, lossless_path


def __bench_cb(f, t):
    print(f"Frame {f}/{t-1}", end="\r")


def video_bench(
    video_path: str,
    encoding: str,
    quality: float,
    metrics: List[str],
    output: str,
    resolution: str,
    keep: bool,
):
    assert quality >= 0.0 and quality <= 100.0
    # Video info
    video_path = get_video(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Prepare workspace
    os.makedirs(TMP_STANDARD_PATH, exist_ok=True)
    # Encode
    if encoding == "avc":
        encoded_path, lossless_path = __encode_avc(video_path, video_name, quality)
    elif encoding == "vp9":
        encoded_path, lossless_path = __encode_vp9(video_path, video_name, quality)
    else:
        raise RuntimeError(f"Unknown encoding {encoding}")
    try:
        # Benching
        device = torch.device("cuda")
        dtype = torch.float
        ref = Y4mBenchInput(video_path, device, dtype)
        dist = Y4mBenchInput(lossless_path, device, dtype)
        print(f"Benching {lossless_path} compared to {video_path}...")
        benchmark = bench(ref, dist, callback=__bench_cb, resolution=resolution, metrics=metrics)
        print()
        # Result
        print(benchmark.to_table())
        if output is not None:
            write_output_file(output, benchmark)
    finally:
        if not keep:
            os.unlink(encoded_path)
            os.unlink(lossless_path)
