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

# don't forget set PYTHON PATH
from kompil.encode import encode
from kompil.utils.ffmpeg import convert_video_to_mjpeg, resize_video, cut_video, get_duration
from kompil.utils.paths import make_dir, clear_dir, PATH_BUILD
from kompil.utils.time import now_str

MJPEG_EXT = ".avi"

BENCH_DIR = os.path.join(PATH_BUILD, f"benchmarks/topologies-{now_str()}")
VIDEO_DIR = os.path.join(PATH_BUILD, "videos")


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
    eval_resolutions: dict,
    eval_durations: list,
    start_time: int = 0,
) -> list:
    assert original_video_path

    make_dir(BENCH_DIR)
    make_dir(VIDEO_DIR)

    original_video_duration = get_duration(original_video_path)
    original_video_file = os.path.basename(original_video_path)
    original_video_name, original_video_ext = os.path.splitext(original_video_file)

    print(f"Generating sub videos from {original_video_file}")

    videos = []

    for duration in eval_durations:
        print(f"Generating videos of duration {duration}s")

        cut_video_path = f"{VIDEO_DIR}/{original_video_name}-{duration}s{original_video_ext}"

        if not os.path.exists(cut_video_path):
            if original_video_duration > duration + start_time:
                cut_video(
                    original_video_path=original_video_path,
                    dest_video_path=cut_video_path,
                    start_time_sec=start_time,
                    length_sec=duration,
                )
            else:
                print(
                    f"    Cannot cut video: starttime + duration ({start_time} + {duration}) > original duration ({original_video_duration})"
                )
                continue

        for width, height in eval_resolutions.items():
            print(f"    Generating video of duration {duration}s at resolution {width}x{height}")

            cut_and_resize_video_path = f"{VIDEO_DIR}/{original_video_name}-{duration}s-{width}x{height}{original_video_ext}"

            if not os.path.exists(cut_and_resize_video_path):
                resize_video(
                    original_video_path=cut_video_path,
                    dest_video_path=cut_and_resize_video_path,
                    width=width,
                    height=height,
                )
            else:
                print(
                    f"        Video for duration {duration} at resolution {width}x{height} already exists !"
                )

            if original_video_ext != MJPEG_EXT:
                print(f"    Generating mjpeg video...")
                final_video_path = (
                    f"{VIDEO_DIR}/{original_video_name}-{duration}s-{width}x{height}{MJPEG_EXT}"
                )
                if not os.path.exists(final_video_path):
                    convert_video_to_mjpeg(
                        original_video_path=cut_and_resize_video_path,
                        dest_video_path=final_video_path,
                    )
                else:
                    print(f"        {MJPEG_EXT} format video already exists !")
            else:
                final_video_path = cut_and_resize_video_path

            videos.append(
                Video(
                    video_path=final_video_path,
                    resolution=(width, height),
                    duration=duration,
                )
            )

            # TODO: should be an option to clear tmp video or not
            # os.unlink(cut_and_resize_video_path)

        # TODO: should be an option to clear tmp video or not
        # os.unlink(cut_video_path)

    return videos


def bench(
    videos: list,
    topologies_path: str,
    autoquit_epoch: int,
    learning_rate: [float, str],
):
    assert videos

    make_dir(BENCH_DIR)

    for video in videos:
        run_build_dir = os.path.join(BENCH_DIR, video.name)

        for topo_filename in os.listdir(topologies_path):
            topo_filepath = os.path.join(topologies_path, topo_filename)

            try:
                model, report = None, None
                model, report = encode(
                    model_name=video.name,
                    video_path=video.path,
                    autoquit_epoch=autoquit_epoch,
                    learning_rate=learning_rate,
                    topology=topo_filepath,
                    loading_mode="auto",  # TODO: should be an option
                    batch_size=32,
                    output_folder=run_build_dir,
                )

                report.save_in(run_build_dir)

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
    decord.bridge.set_bridge("torch")

    durations = [20, 60, 120]
    resolutions = {720: 480, 480: 320}
    # resolutions = {720: 480}
    start_time = 16

    videos = generate_sub_videos(
        original_video_path=argv[0],
        eval_durations=durations,
        eval_resolutions=resolutions,
        start_time=start_time,
    )

    bench(
        videos=videos,
        topologies_path=argv[1],
        autoquit_epoch=2500,
        learning_rate="dynamic",
    )


# Usage : python3 bench_topologies.py /path/to/original/video /path/to/topologies/folder
if __name__ == "__main__":
    main(sys.argv[1:])
