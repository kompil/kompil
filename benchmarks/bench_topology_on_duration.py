import torch
import os
import cv2
import shutil
import time
import sys
import json
import gc
import decord

from shutil import copyfile

from kompil.encode import encode
from kompil.utils.ffmpeg import cut_video, get_duration
from kompil.utils.paths import make_dir, PATH_BUILD
from kompil.utils.time import now_str
from kompil.nn.models.model import model_save

BENCH_DIR = os.path.join(PATH_BUILD, f"benchmarks/topology-duration-{now_str()}")
VIDEOS_DIR = os.path.join(PATH_BUILD, "videos")


class Video:
    def __init__(self, video_path: str, resolution: tuple, duration: int):
        self.resolution = resolution
        self.duration = duration
        self.path = video_path
        self.name, _ = os.path.splitext(os.path.basename(video_path))

    def __str__(self):
        return f"Video<{self.name}, {self.duration}, {self.resolution}>"


def generate_sub_videos(
    original_video_path: str,
    eval_durations: list,
) -> list:
    assert original_video_path

    make_dir(BENCH_DIR)
    make_dir(VIDEOS_DIR)

    original_video_duration = get_duration(original_video_path)
    original_video_file = os.path.basename(original_video_path)
    original_video_name, original_video_ext = os.path.splitext(original_video_file)

    print(f"Generating sub videos from {original_video_file}")

    videos = []

    for duration in eval_durations:
        print(f"Generating videos of duration {duration}s")

        cut_video_path = f"{VIDEOS_DIR}/{original_video_name}-{duration}s{original_video_ext}"

        if not os.path.exists(cut_video_path):
            if original_video_duration > duration:
                cut_video(
                    original_video_path=original_video_path,
                    dest_video_path=cut_video_path,
                    start_time_sec=0,
                    length_sec=duration,
                )
            else:
                print(
                    f"    Cannot cut video: duration ({duration}) > original duration ({original_video_duration})"
                )
                continue

        videos.append(Video(video_path=cut_video_path, duration=duration, resolution=(320, 480)))

    return videos


def bench(
    videos: list,
    topology: str,
    autoquit_epoch: int,
    learning_rate: [float, str],
):
    assert videos

    make_dir(BENCH_DIR)

    for video in videos:
        run_build_dir = os.path.join(BENCH_DIR, video.name)

        try:
            model, report = None, None
            model, report = encode(
                model_name=video.name,
                video_path=video.path,
                autoquit_epoch=autoquit_epoch,
                learning_rate=learning_rate,
                topology=topology,
                loading_mode="auto",
                batch_size=32,
                output_folder=run_build_dir,
                resolution="320p",
                scheduler="vuality",
                no_best=True,
                loss="eps",
            )

            report.save_in(run_build_dir)
            model_save(model, os.path.join(run_build_dir, f"{model.name}.pth"))

            if report.status != "success":
                print("Didn't succeed to encode video, abort further video duration encoding.")
                break

        except Exception as e:
            print(f"Error while encoding {video} : {e}")

        finally:
            print("Free up benchmark memory...")

            if model:
                model.to("cpu")
                del model

            if report:
                del report

            torch.cuda.empty_cache()
            gc.collect()


def main(argv):
    assert len(argv) > 0

    decord.bridge.set_bridge("torch")

    durations = [20, 60, 120, 180, 240]

    videos = generate_sub_videos(
        original_video_path=argv[0],
        eval_durations=durations,
    )

    topology = argv[1] if len(argv) > 1 else "dynamic"

    bench(
        videos=videos,
        topology=topology,
        autoquit_epoch=2500,
        learning_rate="dynamic",
    )


# Usage : python3 bench_topology_on_duration.py /path/to/original/video [/path/to/topo.json, 'dynamic' if None]
if __name__ == "__main__":
    main(sys.argv[1:])
