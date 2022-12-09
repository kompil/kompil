import torch
import os
import kompil.utils.ffmpeg2 as ffmpeg
from kompil.profile.bench import Benchmark, Y4mBenchInput, bench
from kompil.eval import write_output_file

RES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "res", "videos")
SOTA_PATH = os.path.expanduser("~/.kompil/runs/sota/std")

VIDEO_GROUPS = ["beauty", "skate", "tunnel", "witcher"]

VIDEOS = {group: RES_PATH + f"/{group}_320p.y4m" for group in VIDEO_GROUPS}


def bench_y4m(ref: str, dist: str) -> Benchmark:
    def cb(f, t):
        print(f"Frame {f}/{t-1}", end="\r")

    print(f"Benching {dist} compared to {ref}...")

    device = torch.device("cuda")
    dtype = torch.float

    ref_input = Y4mBenchInput(ref, device, dtype)
    dist_input = Y4mBenchInput(dist, device, dtype)

    benchmark = bench(ref_input, dist_input, callback=cb)

    print()
    return benchmark


def run_mpeg2(group: str, video_path: str):
    print("#" * 40)
    print("### running MPGE2")
    codec_path = os.path.join(SOTA_PATH, "MPEG2", group)
    os.makedirs(codec_path, exist_ok=True)

    for quality in range(1, 31, 5):
        quality_path = os.path.join(codec_path, "quality_" + str(quality))
        os.makedirs(quality_path, exist_ok=True)
        # Convert to AVC
        encoded_path = os.path.join(quality_path, f"{group}_q{quality}.m2v")
        ffmpeg.transcode_to_mpeg2(video_path, encoded_path, quality)
        # Return to y4m
        lossless_path = os.path.join(quality_path, f"{group}_q{quality}.y4m")
        ffmpeg.transcode_to_y4m(encoded_path, lossless_path)
        # Bench result
        benchmark = bench_y4m(video_path, lossless_path)
        print(benchmark.to_table())
        # Save benchmark
        bench_path = os.path.join(quality_path, f"{group}_q{quality}.json")
        write_output_file(bench_path, benchmark)

    print()


def run_avc(group: str, video_path: str):
    print("#" * 40)
    print("### running AVC")
    codec_path = os.path.join(SOTA_PATH, "AVC", group)
    os.makedirs(codec_path, exist_ok=True)

    for quality in range(1, 31, 5):
        print("# Quality", quality)
        quality_path = os.path.join(codec_path, "quality_" + str(quality))
        os.makedirs(quality_path, exist_ok=True)
        # Convert to AVC
        encoded_path = os.path.join(quality_path, f"{group}_q{quality}.mp4")
        ffmpeg.transcode_to_h264(video_path, encoded_path, quality=quality)
        # Return to y4m
        lossless_path = os.path.join(quality_path, f"{group}_q{quality}.y4m")
        ffmpeg.transcode_to_y4m(encoded_path, lossless_path)
        # Bench result
        benchmark = bench_y4m(video_path, lossless_path)
        print(benchmark.to_table())
        # Save benchmark
        bench_path = os.path.join(quality_path, f"{group}_q{quality}.json")
        write_output_file(bench_path, benchmark)

    print()


def run_vp9(group: str, video_path: str):
    print("#" * 40)
    print("### running VP9")
    codec_path = os.path.join(SOTA_PATH, "VP9", group)
    os.makedirs(codec_path, exist_ok=True)

    # for quality in range(0, 51, 10):
    for quality in [20]:
        print("# Quality", quality)
        quality_path = os.path.join(codec_path, "quality_" + str(quality))
        os.makedirs(quality_path, exist_ok=True)
        # Convert to VP9
        encoded_path = os.path.join(quality_path, f"{group}_q{quality}.webm")
        ffmpeg.transcode_to_vp9(video_path, encoded_path, quality=quality)
        # Return to y4m
        lossless_path = os.path.join(quality_path, f"{group}_q{quality}.y4m")
        ffmpeg.transcode_to_y4m(encoded_path, lossless_path)
        # Bench result
        benchmark = bench_y4m(video_path, lossless_path)
        print(benchmark.to_table())
        # Save benchmark
        bench_path = os.path.join(quality_path, f"{group}_q{quality}.json")
        write_output_file(bench_path, benchmark)

    print()


def one_video(group: str):
    video_path = os.path.join(RES_PATH, VIDEOS[group])
    print(video_path)

    run_mpeg2(group, video_path)
    run_avc(group, video_path)
    run_vp9(group, video_path)


def main():
    for group in VIDEO_GROUPS:
        one_video(group)


if __name__ == "__main__":
    main()
